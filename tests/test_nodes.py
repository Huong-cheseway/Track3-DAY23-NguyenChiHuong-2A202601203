from types import SimpleNamespace
from typing import cast

import pytest
from langchain_core.language_models import BaseChatModel

from langgraph_agent_lab import nodes
from langgraph_agent_lab.nodes import (
    ClassificationResult,
    answer_node,
    approval_node,
    ask_clarification_node,
    classify_node,
    dead_letter_node,
    evaluate_node,
    finalize_node,
    retry_or_fallback_node,
    risky_action_node,
    tool_node,
)
from langgraph_agent_lab.state import AgentState


class FakeStructuredClassifier:
    def __init__(self, result: ClassificationResult) -> None:
        self.result = result
        self.prompt = ""
        self.schema: type[ClassificationResult] | None = None

    def with_structured_output(
        self, schema: type[ClassificationResult]
    ) -> "FakeStructuredClassifier":
        self.schema = schema
        return self

    def invoke(self, prompt: str) -> ClassificationResult:
        self.prompt = prompt
        return self.result


class FakeAnswerClient:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompt = ""

    def invoke(self, prompt: str) -> SimpleNamespace:
        self.prompt = prompt
        return SimpleNamespace(content=self.answer)


def _strings(update: dict[str, object], key: str) -> list[str]:
    return cast(list[str], update[key])


def _events(update: dict[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], update["events"])


@pytest.mark.parametrize(
    ("route", "expected_risk"),
    [
        ("simple", "low"),
        ("tool", "low"),
        ("missing_info", "low"),
        ("error", "low"),
        ("risky", "high"),
    ],
)
def test_classify_node_uses_structured_output(
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    expected_risk: str,
) -> None:
    result = ClassificationResult.model_validate({"route": route, "reason": "Test reason"})
    fake_llm = FakeStructuredClassifier(result)
    monkeypatch.setattr(nodes, "get_llm", lambda temperature=0.0: cast(BaseChatModel, fake_llm))

    update = classify_node({"query": "Please help with this request"})

    assert update["route"] == route
    assert update["risk_level"] == expected_risk
    assert fake_llm.schema is ClassificationResult
    assert "risky > tool > missing_info > error > simple" in fake_llm.prompt
    assert "A how-to question is not risky" in fake_llm.prompt
    assert "Do not infer a\n  risky action" in fake_llm.prompt
    assert "Please help with this request" in fake_llm.prompt
    assert _events(update)[0]["node"] == "classify"


def test_tool_node_simulates_transient_error_then_recovery() -> None:
    failed = tool_node({"route": "error", "attempt": 1, "query": "Timeout"})
    recovered = tool_node({"route": "error", "attempt": 2, "query": "Timeout"})

    assert "ERROR" in _strings(failed, "tool_results")[0]
    assert "recovered successfully" in _strings(recovered, "tool_results")[0]
    assert _events(failed)[0]["node"] == "tool"


def test_tool_node_requires_approval_for_risky_action() -> None:
    blocked = tool_node({"route": "risky", "query": "Delete account", "approval": None})
    approved = tool_node(
        {
            "route": "risky",
            "query": "Delete account",
            "proposed_action": "Mock account deletion",
            "approval": {"approved": True},
        }
    )

    assert "ERROR" in _strings(blocked, "tool_results")[0]
    assert "Approved mock action completed" in _strings(approved, "tool_results")[0]


@pytest.mark.parametrize(
    ("tool_results", "expected"),
    [([], "needs_retry"), (["ERROR: unavailable"], "needs_retry"), (["Lookup OK"], "success")],
)
def test_evaluate_node_sets_retry_gate(tool_results: list[str], expected: str) -> None:
    update = evaluate_node({"tool_results": tool_results})

    assert update["evaluation_result"] == expected
    assert _events(update)[0]["node"] == "evaluate"


def test_answer_node_uses_llm_with_grounded_context(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_llm = FakeAnswerClient("Your order is being processed.")
    monkeypatch.setattr(nodes, "get_llm", lambda temperature=0.0: cast(BaseChatModel, fake_llm))
    state: AgentState = {
        "query": "Where is order 123?",
        "route": "tool",
        "tool_results": ["Order 123 status: processing"],
        "approval": None,
    }

    update = answer_node(state)

    assert update["final_answer"] == "Your order is being processed."
    assert "Where is order 123?" in fake_llm.prompt
    assert "Order 123 status: processing" in fake_llm.prompt
    assert _events(update)[0]["node"] == "answer"


def test_ask_clarification_node_handles_missing_information() -> None:
    update = ask_clarification_node({"query": "Can you fix it?", "route": "missing_info"})

    question = cast(str, update["pending_question"])
    assert "Can you fix it?" in question
    assert "error message" in question
    assert update["final_answer"] == question


def test_ask_clarification_node_handles_rejected_action() -> None:
    update = ask_clarification_node(
        {"query": "Delete account", "route": "risky", "approval": {"approved": False}}
    )

    assert "not approved" in cast(str, update["pending_question"])


def test_risky_action_node_prepares_reviewable_action() -> None:
    update = risky_action_node({"query": "Refund the customer", "route": "risky"})

    proposal = cast(str, update["proposed_action"])
    assert "Refund the customer" in proposal
    assert "approval is required" in proposal


def test_approval_node_uses_mock_approval_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGGRAPH_INTERRUPT", "false")

    update = approval_node({"proposed_action": "Mock refund"})

    decision = cast(dict[str, object], update["approval"])
    assert decision["approved"] is True
    assert decision["reviewer"] == "mock-reviewer"
    assert _events(update)[0]["node"] == "approval"


def test_retry_node_increments_without_mutating_input() -> None:
    state: AgentState = {"attempt": 1, "tool_results": ["ERROR: timeout"]}

    update = retry_or_fallback_node(state)

    assert state["attempt"] == 1
    assert update["attempt"] == 2
    assert "ERROR: timeout" in _strings(update, "errors")[0]
    assert _events(update)[0]["node"] == "retry"


def test_dead_letter_node_preserves_route_and_explains_failure() -> None:
    update = dead_letter_node({"route": "error", "attempt": 1, "max_attempts": 1})

    assert "route" not in update
    assert "1 of 1 allowed attempts" in cast(str, update["final_answer"])
    assert _events(update)[0]["node"] == "dead_letter"


def test_finalize_node_emits_terminal_event() -> None:
    update = finalize_node({})

    assert _events(update)[0]["node"] == "finalize"
