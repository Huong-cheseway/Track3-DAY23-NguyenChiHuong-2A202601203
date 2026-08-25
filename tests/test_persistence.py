from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph_agent_lab import nodes
from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.nodes import ClassificationResult
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state


class FakeSimpleLlm:
    def __init__(self) -> None:
        self.structured = False

    def with_structured_output(self, schema: type[ClassificationResult]) -> "FakeSimpleLlm":
        assert schema is ClassificationResult
        self.structured = True
        return self

    def invoke(self, prompt: str) -> ClassificationResult | SimpleNamespace:
        if self.structured:
            return ClassificationResult(route="simple", reason="persistence test")
        return SimpleNamespace(content="Persisted grounded response.")


def test_build_checkpointer_supports_none_and_memory() -> None:
    assert build_checkpointer("none") is None
    assert isinstance(build_checkpointer("memory"), MemorySaver)


def test_build_checkpointer_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unknown checkpointer kind"):
        build_checkpointer("unsupported")


def test_sqlite_checkpoint_survives_connection_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        nodes,
        "get_llm",
        lambda temperature=0.0: cast(BaseChatModel, FakeSimpleLlm()),
    )
    database_path = tmp_path / "nested" / "checkpoints.db"
    scenario = Scenario(
        id="sqlite-recovery", query="How can I get help?", expected_route=Route.SIMPLE
    )
    state = initial_state(scenario)
    config: RunnableConfig = {"configurable": {"thread_id": state["thread_id"]}}

    first_saver = cast(SqliteSaver, build_checkpointer("sqlite", str(database_path)))
    try:
        first_graph = build_graph(first_saver)
        result = first_graph.invoke(state, config=config)
        history = list(first_graph.get_state_history(config))

        assert result["final_answer"] == "Persisted grounded response."
        assert history
        assert first_saver.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        first_saver.conn.close()

    second_saver = cast(SqliteSaver, build_checkpointer("sqlite", str(database_path)))
    try:
        second_graph = build_graph(second_saver)
        recovered = second_graph.get_state(config)
        recovered_events = recovered.values.get("events", [])

        assert recovered.values["final_answer"] == "Persisted grounded response."
        assert any(event.get("node") == "finalize" for event in recovered_events)
        assert database_path.exists()
    finally:
        second_saver.conn.close()
