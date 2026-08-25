"""Offline end-to-end smoke tests for graph wiring and route behavior."""

from types import SimpleNamespace
from typing import cast

import pytest
from langchain_core.language_models import BaseChatModel

from langgraph_agent_lab import nodes
from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.metrics import metric_from_state
from langgraph_agent_lab.nodes import ClassificationResult
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.scenarios import load_scenarios
from langgraph_agent_lab.state import Route, Scenario, initial_state


class FakeWorkflowLlm:
    """Return structured classifications and grounded text without an API call."""

    def __init__(self, route: Route) -> None:
        self.route = route
        self.structured = False

    def with_structured_output(self, schema: type[ClassificationResult]) -> "FakeWorkflowLlm":
        assert schema is ClassificationResult
        self.structured = True
        return self

    def invoke(self, prompt: str) -> ClassificationResult | SimpleNamespace:
        if self.structured:
            return ClassificationResult.model_validate(
                {"route": self.route.value, "reason": "offline graph test"}
            )
        return SimpleNamespace(content="Grounded mock support response.")


def _use_fake_llm(monkeypatch: pytest.MonkeyPatch, route: Route) -> None:
    monkeypatch.setattr(
        nodes,
        "get_llm",
        lambda temperature=0.0: cast(BaseChatModel, FakeWorkflowLlm(route)),
    )


@pytest.mark.parametrize(
    ("query", "expected_route"),
    [
        ("How do I reset my password?", Route.SIMPLE.value),
        ("Please lookup order status for order 123", Route.TOOL.value),
        ("Refund this customer", Route.RISKY.value),
        ("Can you fix it?", Route.MISSING_INFO.value),
        ("Timeout failure while processing", Route.ERROR.value),
    ],
)
def test_graph_runs_and_routes_correctly(
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    expected_route: str,
) -> None:
    route = Route(expected_route)
    _use_fake_llm(monkeypatch, route)
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    scenario = Scenario(id="smoke", query=query, expected_route=route)
    state = initial_state(scenario)
    result = graph.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})
    assert result["route"] == expected_route
    assert result.get("final_answer") or result.get("pending_question")


def test_all_sample_scenarios_terminate_with_expected_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise all seven scenarios, including approval, retry, and dead letter."""
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    for scenario in load_scenarios("data/sample/scenarios.jsonl"):
        _use_fake_llm(monkeypatch, scenario.expected_route)
        state = initial_state(scenario)
        result = graph.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})
        events = result.get("events", [])
        visited = [event.get("node") for event in events]
        metric = metric_from_state(
            result,
            scenario.expected_route.value,
            scenario.requires_approval,
        )

        assert metric.success, scenario.id
        assert "finalize" in visited, scenario.id
        if scenario.requires_approval:
            assert "approval" in visited, scenario.id
        if scenario.id == "S05_error":
            assert visited.count("retry") == 2
        if scenario.id == "S07_dead_letter":
            assert visited.count("retry") == 1
            assert "dead_letter" in visited
