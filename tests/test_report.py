from pathlib import Path

from langgraph_agent_lab.metrics import MetricsReport, ScenarioMetric
from langgraph_agent_lab.report import render_report, write_report


def _sample_metrics(*, resume_success: bool = True) -> MetricsReport:
    return MetricsReport(
        total_scenarios=2,
        success_rate=0.5,
        avg_nodes_visited=4.5,
        total_retries=1,
        total_interrupts=1,
        resume_success=resume_success,
        scenario_metrics=[
            ScenarioMetric(
                scenario_id="S01|simple",
                success=True,
                expected_route="simple",
                actual_route="simple",
                nodes_visited=4,
                retry_count=0,
                interrupt_count=0,
            ),
            ScenarioMetric(
                scenario_id="S02",
                success=False,
                expected_route="tool",
                actual_route=None,
                nodes_visited=5,
                retry_count=1,
                interrupt_count=1,
                approval_required=True,
                approval_observed=True,
                errors=["timeout"],
            ),
        ],
    )


def test_render_report_contains_metrics_architecture_and_scenarios() -> None:
    report = render_report(_sample_metrics())

    assert "# Day 08 Lab Report" in report
    assert "Success rate | 50.00%" in report
    assert "11-node LangGraph state machine" in report
    assert "build_graph().get_graph().draw_mermaid()" in report
    assert "```mermaid" in report
    assert "risky_action --> approval" in report
    assert "S01\\|simple" in report
    assert "| S02 | tool | — | No | 1 | 1 | Yes | 0 |" in report
    assert "Recovery evidence observed | Yes" in report
    assert "53 passed" in report


def test_render_report_does_not_claim_missing_recovery_evidence() -> None:
    report = render_report(_sample_metrics(resume_success=False))

    assert "Recovery evidence observed | No" in report
    assert "did not record successful checkpoint recovery evidence" in report


def test_write_report_creates_parent_directory(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "lab_report.md"

    write_report(_sample_metrics(), output)

    assert output.is_file()
    assert output.read_text(encoding="utf-8").startswith("# Day 08 Lab Report")
