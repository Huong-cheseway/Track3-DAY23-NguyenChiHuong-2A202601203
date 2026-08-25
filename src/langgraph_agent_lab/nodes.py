"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import json
import os
from typing import Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, ApprovalDecision, make_event


class ClassificationResult(BaseModel):
    """Structured output returned by the intent-classification LLM call."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"]
    reason: str = Field(min_length=1, description="Brief reason for selecting this route")


def _response_text(response: object) -> str:
    """Normalize text content returned by supported LangChain chat models."""
    content: object = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip()


def _approval_is_granted(state: AgentState) -> bool:
    approval = state.get("approval") or {}
    return bool(approval.get("approved", False))


def _parse_approval(value: object) -> ApprovalDecision:
    if isinstance(value, bool):
        return ApprovalDecision(approved=value, reviewer="human-reviewer")
    if isinstance(value, dict):
        return ApprovalDecision.model_validate(value)
    return ApprovalDecision(comment="Invalid or missing human approval response")


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict[str, object]:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── WORKFLOW NODES ─────────────────────────────────────────────────


def classify_node(state: AgentState) -> dict[str, object]:
    """Classify the query into a route using an LLM.

    *** MUST use a real LLM call — keyword-only heuristics will lose points. ***

    Use .with_structured_output() or equivalent to get reliable enum classification.
    The LLM should classify into one of: simple, tool, missing_info, risky, error.

    Hints:
    - See llm.py for the get_llm() helper
    - Use Pydantic model or TypedDict with .with_structured_output()
    - Set risk_level to "high" for risky routes, "low" otherwise
    - Priority guide: risky > tool > missing_info > error > simple

    Return: {"route": str, "risk_level": str, "events": [make_event(...)]}
    """
    query = state.get("query", "").strip()
    prompt = f"""You classify customer-support requests for a LangGraph workflow.

Choose exactly one route:
- risky: the customer clearly asks this agent/workflow to execute an external side effect,
  including issuing refunds, deleting records, cancelling services, sending messages, or changing
  customer/account/order data. A how-to question is not risky merely because the procedure it
  describes could change something.
- tool: a read-only lookup, tracking request, search, or request for external information.
- missing_info: too vague or incomplete to identify the issue or requested action. Do not infer a
  risky action from a generic verb or pronoun when the affected object and desired outcome are not
  stated.
- error: a reported technical/system failure such as a timeout, crash, or unavailable service.
- simple: a general support question or instructions the customer will carry out themselves,
  answerable without this workflow using tools or performing side effects.

When multiple routes seem applicable, apply this priority:
risky > tool > missing_info > error > simple.
Apply that priority only after a route's definition is actually satisfied; priority must not turn
an informational or underspecified request into a risky action.

Classify by intent and meaning. Do not rely on scenario identifiers or exact-example matching.

Customer request:
{query}
"""
    structured_llm = get_llm(temperature=0.0).with_structured_output(ClassificationResult)
    payload = structured_llm.invoke(prompt)
    classification = (
        payload
        if isinstance(payload, ClassificationResult)
        else ClassificationResult.model_validate(payload)
    )
    risk_level = "high" if classification.route == "risky" else "low"
    return {
        "route": classification.route,
        "risk_level": risk_level,
        "events": [
            make_event(
                "classify",
                "completed",
                classification.reason,
                route=classification.route,
                risk_level=risk_level,
            )
        ],
    }


def tool_node(state: AgentState) -> dict[str, object]:
    """Execute a mock tool call.

    Simulate transient failures for error-route scenarios to test retry loops.

    Requirements:
    - Read current attempt count from state
    - If route is "error" and attempt < 2: return error result (string containing "ERROR")
    - Otherwise: return a mock success result string
    - Append result to tool_results list

    Return: {"tool_results": [result_string], "events": [make_event(...)]}
    """
    route = state.get("route", "")
    attempt = state.get("attempt", 0)
    query = state.get("query", "")

    if route == "error" and attempt < 2:
        result = f"ERROR: transient support tool failure on attempt {attempt}"
        event_type = "failed"
    elif route == "risky" and not _approval_is_granted(state):
        result = "ERROR: risky action was blocked because approval is missing"
        event_type = "blocked"
    elif route == "risky":
        action = state.get("proposed_action") or query
        result = f"Approved mock action completed: {action}"
        event_type = "completed"
    elif route == "error":
        result = f"Mock support tool recovered successfully on attempt {attempt}"
        event_type = "completed"
    else:
        result = f"Mock lookup completed for request: {query}"
        event_type = "completed"

    return {
        "tool_results": [result],
        "events": [
            make_event(
                "tool",
                event_type,
                result,
                attempt=attempt,
            )
        ],
    }


def evaluate_node(state: AgentState) -> dict[str, object]:
    """Evaluate tool results — the retry-loop gate.

    Check whether the latest tool result is satisfactory or needs retry.

    SHOULD use LLM-as-judge for bonus points. Heuristic (e.g., check for "ERROR" substring)
    is acceptable for base score.

    Requirements:
    - Read the latest entry from tool_results
    - Set evaluation_result to "needs_retry" or "success"
    - This field drives route_after_evaluate conditional edge

    Note: You may need to add 'evaluation_result' to AgentState if not present.

    Return: {"evaluation_result": str, "events": [make_event(...)]}
    """
    tool_results = state.get("tool_results", [])
    latest_result = tool_results[-1] if tool_results else ""
    needs_retry = not latest_result or "ERROR" in latest_result.upper()
    evaluation_result = "needs_retry" if needs_retry else "success"
    event_type = "retry_required" if needs_retry else "completed"
    return {
        "evaluation_result": evaluation_result,
        "events": [
            make_event(
                "evaluate",
                event_type,
                f"tool result evaluation: {evaluation_result}",
            )
        ],
    }


def answer_node(state: AgentState) -> dict[str, object]:
    """Generate a final response using an LLM.

    *** MUST use a real LLM call — hardcoded strings will lose points. ***

    The LLM should generate a helpful response grounded in available context:
    - tool_results (if any)
    - approval decision (if risky route)
    - original query

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    context = {
        "customer_request": state.get("query", ""),
        "route": state.get("route", ""),
        "tool_results": state.get("tool_results", []),
        "proposed_action": state.get("proposed_action"),
        "approval": state.get("approval"),
    }
    prompt = f"""You are a careful customer-support assistant.

Write a concise, helpful final response using only the supplied workflow context. Treat tool
results as the available source of truth. Do not invent order details, customer data, actions,
or outcomes. Never claim that a risky action was completed unless the context contains an
approved decision and a successful tool result. If information is unavailable, say so clearly.

Workflow context:
{json.dumps(context, ensure_ascii=False, default=str)}
"""
    response = get_llm(temperature=0.0).invoke(prompt)
    answer = _response_text(response)
    if not answer:
        raise RuntimeError("The LLM returned an empty final answer")
    return {
        "final_answer": answer,
        "events": [make_event("answer", "completed", "grounded response generated")],
    }


def ask_clarification_node(state: AgentState) -> dict[str, object]:
    """Ask for missing information instead of hallucinating.

    Generate a specific clarification question based on the vague/incomplete query.

    Note: You may need to add 'pending_question' to AgentState if not present.

    Return: {"pending_question": str, "final_answer": str, "events": [make_event(...)]}
    """
    query = state.get("query", "").strip()
    approval = state.get("approval")
    if state.get("route") == "risky" and approval is not None and not _approval_is_granted(state):
        question = (
            "The proposed action was not approved. What non-destructive alternative would you "
            "like us to take, or can you provide updated authorization?"
        )
    else:
        question = (
            f'To help with "{query}", please provide the affected product or service, what you '
            "were trying to do, and any error message or relevant identifier."
        )
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "clarification requested")],
    }


def risky_action_node(state: AgentState) -> dict[str, object]:
    """Prepare a risky action for human approval.

    Describe the proposed action and why it requires approval.

    Note: You may need to add 'proposed_action' to AgentState if not present.

    Return: {"proposed_action": str, "events": [make_event(...)]}
    """
    query = state.get("query", "").strip()
    proposed_action = (
        f"Proposed customer-impacting action: {query}. Human approval is required before the "
        "mock tool can execute this request."
    )
    return {
        "proposed_action": proposed_action,
        "events": [
            make_event(
                "risky_action",
                "approval_required",
                "risky action prepared for review",
            )
        ],
    }


def approval_node(state: AgentState) -> dict[str, object]:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for real HITL.

    Return an approval decision and an audit event.
    """
    decision = ApprovalDecision(
        approved=True,
        reviewer="mock-reviewer",
        comment="Approved automatically for reproducible lab execution",
    )
    interrupt_enabled = os.getenv("LANGGRAPH_INTERRUPT", "false").casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if interrupt_enabled:
        from langgraph.types import interrupt

        resumed_value: object = interrupt(
            {
                "question": "Approve this customer-impacting action?",
                "proposed_action": state.get("proposed_action", ""),
            }
        )
        decision = _parse_approval(resumed_value)

    event_type = "approved" if decision.approved else "rejected"
    return {
        "approval": decision.model_dump(),
        "events": [
            make_event(
                "approval",
                event_type,
                decision.comment or "approval decision recorded",
                reviewer=decision.reviewer,
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict[str, object]:
    """Record a retry attempt.

    Increment the attempt counter and log the transient failure.

    Requirements:
    - Read current attempt from state, increment by 1
    - Add an error message to errors list
    - Return updated attempt count

    Return: {"attempt": int, "errors": [str], "events": [make_event(...)]}
    """
    attempt = state.get("attempt", 0) + 1
    tool_results = state.get("tool_results", [])
    latest_result = tool_results[-1] if tool_results else "no tool result available"
    error = f"Retry attempt {attempt}: {latest_result}"
    return {
        "attempt": attempt,
        "errors": [error],
        "events": [make_event("retry", "scheduled", error, attempt=attempt)],
    }


def dead_letter_node(state: AgentState) -> dict[str, object]:
    """Handle unresolvable failures after max retries exceeded.

    This is the third layer: retry → fallback → dead letter.
    Log the failure and set a final_answer explaining that the request could not be completed.

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    final_answer = (
        "We could not complete this request after "
        f"{attempt} of {max_attempts} allowed attempts. The issue has been recorded for review."
    )
    return {
        "final_answer": final_answer,
        "events": [
            make_event(
                "dead_letter",
                "failed",
                "retry limit reached; request escalated",
                attempt=attempt,
                max_attempts=max_attempts,
            )
        ],
    }


def finalize_node(state: AgentState) -> dict[str, object]:
    """Emit a final audit event. All routes must pass through here before END.

    Return: {"events": [make_event("finalize", "completed", "workflow finished")]}
    """
    return {
        "events": [make_event("finalize", "completed", "workflow finished")],
    }
