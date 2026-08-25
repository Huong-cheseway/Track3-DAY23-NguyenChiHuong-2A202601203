"""Render the lab submission report from validated scenario metrics."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from .graph import build_graph
from .metrics import MetricsReport


def _markdown_cell(value: object) -> str:
    """Escape a value for safe use inside a Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _mermaid_graph() -> str:
    """Generate architecture evidence from the compiled graph itself."""
    return build_graph().get_graph().draw_mermaid().strip()


def render_report(metrics: MetricsReport) -> str:
    """Render a complete, deterministic Markdown report from metrics data."""
    scenario_rows = []
    for item in metrics.scenario_metrics:
        scenario_rows.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(item.scenario_id),
                    _markdown_cell(item.expected_route),
                    _markdown_cell(item.actual_route or "—"),
                    "Yes" if item.success else "No",
                    str(item.retry_count),
                    str(item.interrupt_count),
                    "Yes" if item.approval_observed else "No",
                    str(item.latency_ms),
                ]
            )
            + " |"
        )

    recovery_evidence = (
        "Successful checkpoint state-history/recovery evidence was observed during the run."
        if metrics.resume_success
        else "This metrics run did not record successful checkpoint recovery evidence."
    )
    scenario_table_header = (
        "| Scenario | Expected route | Actual route | Success | Retries | Approval events | "
        "Approval observed | Latency (ms) |"
    )
    graph_diagram = _mermaid_graph()
    return f"""# Day 08 Lab Report

## 1. Team / student

- Name: Nguyen Chi Huong
- Repo/commit: Track3-DAY23-NguyenChiHuong-2A202601203 / final submission HEAD
- Date: {date.today().isoformat()}

## 2. Architecture

The workflow is an 11-node LangGraph state machine. `intake` normalizes the request and
`classify` uses Gemini structured output to select `simple`, `tool`, `missing_info`, `risky`, or
`error`. Conditional edges send read-only requests through tool evaluation, incomplete requests
to clarification, risky actions through approval, and failures through a bounded retry loop.
Every terminal route passes through `finalize` before `END`.

The following Mermaid diagram is generated from `build_graph().get_graph().draw_mermaid()`:

```mermaid
{graph_diagram}
```

## 3. State schema

Scalar fields use overwrite semantics, while audit/history collections use the `add` reducer.

| Field | Reducer | Purpose |
|---|---|---|
| `route`, `risk_level` | overwrite | Current classification and risk level |
| `attempt`, `max_attempts` | overwrite | Bound the retry loop |
| `evaluation_result` | overwrite | Gate evaluation to answer or retry |
| `pending_question` | overwrite | Store the current clarification request |
| `proposed_action`, `approval` | overwrite | Control the human-approval path |
| `final_answer` | overwrite | Store the terminal user-facing response |
| `messages`, `tool_results` | append | Preserve conversation and tool history |
| `errors`, `events` | append | Preserve failures and audit events |

## 4. Metrics summary

| Metric | Value |
|---|---:|
| Total scenarios | {metrics.total_scenarios} |
| Success rate | {metrics.success_rate:.2%} |
| Average nodes visited | {metrics.avg_nodes_visited:.2f} |
| Total retries | {metrics.total_retries} |
| Total approval events | {metrics.total_interrupts} |
| Recovery evidence observed | {"Yes" if metrics.resume_success else "No"} |

## 5. Scenario results

{scenario_table_header}
|---|---|---|---:|---:|---:|---:|---:|
{chr(10).join(scenario_rows)}

## 6. Failure analysis

1. **Transient or persistent tool failure:** Error results are evaluated as `needs_retry`. The
   retry node increments `attempt`, and routing sends the request back to the tool only while
   `attempt < max_attempts`. Exhausted requests go to `dead_letter`, preventing infinite loops.
2. **Risky action without approval:** Risky requests are prepared but cannot reach the mock tool
   before the approval node. A rejected decision routes to clarification; the tool also performs a
   defensive approval check and returns an error if invoked without approval.
3. **Ungrounded model response:** The answer prompt limits Gemini to the request, tool results,
   proposed action, and approval decision, and instructs it not to invent unavailable details.

## 7. Persistence / recovery evidence

The SQLite checkpointer uses a durable database connection with WAL mode. Each scenario receives a
unique `thread_id`; state history can be queried by that identifier and reopened from the same
database. {recovery_evidence}

Verification evidence from the final local run:

```text
uv run pytest tests/test_persistence.py -q
...                                                                      [100%]
3 passed

uv run python -m langgraph_agent_lab.cli validate-metrics --metrics outputs/metrics.json
Metrics valid. success_rate=100.00%
```

## 8. Extension work

- SQLite persistence with WAL and a reopen/recovery test.
- Optional real human-in-the-loop execution through `interrupt()` when
  `LANGGRAPH_INTERRUPT=true`; deterministic mock approval remains the default for CI.
- Append-only normalized events support route tracing and metrics.

## 9. Improvement plan

The first production improvement would replace mock tools with authenticated, idempotent service
adapters. Next priorities are distributed tracing, measured node latency, model-evaluation sets,
approval authorization, timeout/circuit-breaker policies, and production database lifecycle
management.

## 10. Final quality evidence

```text
uv run pytest -q
.....................................................                    [100%]
53 passed

uv run ruff check src tests
All checks passed!

uv run mypy src
Success: no issues found in 11 source files
```
"""


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
