"""Generate both Lakeview dashboards from their annotated SQL files.

    sql/07_drift_dashboard.sql            -> fraud_drift_monitoring.lvdash.json
    sql/09_fraud_impact_dashboard.sql     -> fraud_business_impact.lvdash.json
    sql/11_fraud_operations_dashboard.sql -> fraud_operations_replay.lvdash.json

Why generate rather than hand-author the JSON: the SQL file is the reviewed,
commented artefact -- it is what the thesis cites and what a reader reads. If
the dashboard carried its own copy of the queries, the two would diverge the
first time a threshold changed. So the queries are parsed out of the SQL file
verbatim and the JSON holds only *layout*.

    python dashboards/build_lvdash.py            # rewrite both .lvdash.json
    python dashboards/build_lvdash.py --check    # fail if either is out of date

Import into a workspace (normally the bundle does this for you):
    databricks lakeview create --json @dashboards/fraud_drift_monitoring.lvdash.json
or SQL > Dashboards > ... > Import dashboard from file.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
SQL_PATH = HERE.parent / "sql" / "07_drift_dashboard.sql"
OUT_PATH = HERE / "fraud_drift_monitoring.lvdash.json"

IMPACT_SQL_PATH = HERE.parent / "sql" / "09_fraud_impact_dashboard.sql"
IMPACT_OUT_PATH = HERE / "fraud_business_impact.lvdash.json"

OPS_SQL_PATH = HERE.parent / "sql" / "11_fraud_operations_dashboard.sql"
OPS_OUT_PATH = HERE / "fraud_operations_replay.lvdash.json"

# Query id in the SQL file -> dataset name used throughout the layout below.
DATASETS = {
    "Q1": ("q1_volume", "Q1 · transaction volume"),
    "Q2": ("q2_score_histogram", "Q2 · fraud-score distribution"),
    "Q3": ("q3_feature_psi", "Q3 · feature drift (PSI)"),
    "Q3b": ("q3b_missing_rate", "Q3b · missing-value rate change"),
    "Q4": ("q4_prediction_drift", "Q4 · predicted-positive rate"),
    "Q5": ("q5_performance", "Q5 · precision and recall"),
    "Q6": ("q6_rag_indicator", "Q6 · red / amber / green indicator"),
    "Q7": ("q7_data_quality", "Q7 · data-quality summary"),
    "Q8": ("q8_feature_contract", "Q8 · model and feature contract"),
}

# Same idea for the business impact dashboard, whose blocks are lettered by
# dashboard section (A = KPI tiles, B = trade-off, ... F = recommendation).
IMPACT_DATASETS = {
    "A1": ("a1_net_savings", "A1 · estimated net savings"),
    "A2": ("a2_kpi_tiles", "A2 · executive KPI tiles"),
    "A3": ("a3_assumptions", "A3 · assumptions in force"),
    "B1": ("b1_threshold_tradeoff", "B1 · savings and costs by threshold"),
    "C1": ("c1_caught_vs_missed", "C1 · fraud caught vs missed"),
    "D1": ("d1_review_workload", "D1 · review workload"),
    "E1": ("e1_quality_vs_value", "E1 · model quality vs business value"),
    "F1": ("f1_recommendation", "F1 · business recommendation"),
    "F2": ("f2_neighbouring", "F2 · neighbouring thresholds"),
}

# The operational replay -- the only dashboard whose x-axis is clock time. Its
# blocks are lettered T/U/V/W/X so the section letters cannot collide with the
# impact dashboard's A-F if the two files are ever parsed together.
OPS_DATASETS = {
    "T1": ("t1_since_golive", "T1 · since go-live roll-up"),
    "T2": ("t2_savings_ratio", "T2 · savings per unit of operational cost"),
    "U1": ("u1_savings_vs_cost", "U1 · savings vs operational cost"),
    "U2": ("u2_cumulative", "U2 · cumulative net savings"),
    "U3": ("u3_daily_rollup", "U3 · daily roll-up"),
    "V1": ("v1_caught_vs_missed", "V1 · fraud caught vs missed (count)"),
    "V2": ("v2_value_caught_vs_missed", "V2 · fraud caught vs missed (value)"),
    "W1": ("w1_alerts_vs_capacity", "W1 · alerts vs review capacity"),
    "W2": ("w2_hour_profile", "W2 · alert load by hour of day"),
    "X1": ("x1_quality_over_time", "X1 · precision and recall over time"),
    "X2": ("x2_worst_periods", "X2 · worst operating periods"),
}

HEADER_RE = re.compile(r"^--\s*(Q\d+b?)\s*·", re.MULTILINE)
IMPACT_HEADER_RE = re.compile(r"^--\s*([A-F]\d+)\s*·", re.MULTILINE)
OPS_HEADER_RE = re.compile(r"^--\s*([TUVWX]\d+)\s*·", re.MULTILINE)


def parse_queries(
    sql_text: str,
    datasets: dict | None = None,
    header_re: re.Pattern | None = None,
    source: pathlib.Path | None = None,
) -> dict[str, str]:
    """Pull each `-- Qn ·` block's statement out of the annotated SQL file.

    Everything before the first non-comment line of a block is documentation and
    is dropped: the dashboard carries the prose in its own text widgets, and
    duplicating it inside the query body would make the tile header unreadable.
    Trailing comment banners are dropped too -- they belong to the *next* block
    and would otherwise ride along inside this one's statement.

    Defaults target the drift dashboard; pass the `IMPACT_*` trio for the
    business impact one.
    """
    datasets = DATASETS if datasets is None else datasets
    header_re = HEADER_RE if header_re is None else header_re
    source = SQL_PATH if source is None else source

    marks = [(m.group(1), m.start()) for m in header_re.finditer(sql_text)]
    if not marks:
        raise SystemExit(f"no query headers found in {source}")

    out: dict[str, str] = {}
    for i, (qid, start) in enumerate(marks):
        end = marks[i + 1][1] if i + 1 < len(marks) else len(sql_text)
        block = sql_text[start:end]
        lines = block.splitlines()
        body: list[str] = []
        started = False
        for line in lines:
            stripped = line.strip()
            if not started:
                # Skip the comment banner; the statement starts at the first
                # line that is neither a comment nor blank.
                if stripped.startswith("--") or not stripped:
                    continue
                started = True
            body.append(line)
        # Drop the separator/banner comments that lead into the NEXT block.
        while body and (not body[-1].strip() or body[-1].lstrip().startswith("--")):
            body.pop()
        statement = "\n".join(body).strip()
        # Lakeview datasets hold a single statement; a trailing `;` is rejected.
        statement = statement.rstrip().rstrip(";").rstrip()
        if not statement:
            raise SystemExit(f"{qid}: no statement found under the comment banner")
        out[qid] = statement

    missing = [q for q in datasets if q not in out]
    if missing:
        raise SystemExit(f"queries missing from {source.name}: {missing}")
    return {qid: out[qid] for qid in datasets}


# --------------------------------------------------------------------------- #
# Widget builders. Lakeview's grid is TWELVE columns wide; `height` is in rows.
# Every row should sum to width 12 with no gaps.
# --------------------------------------------------------------------------- #
def _query(dataset: str, fields: list[str]) -> list[dict]:
    return [{
        "name": "main_query",
        "query": {
            "datasetName": dataset,
            "fields": [{"name": f, "expression": f"`{f}`"} for f in fields],
            "disaggregated": True,
        },
    }]


def _frame(title: str, description: str) -> dict:
    return {
        "title": title,
        "description": description,
        "showTitle": True,
        "showDescription": True,
    }


def text(name: str, markdown: str, pos: tuple) -> dict:
    x, y, w, h = pos
    return {
        "widget": {"name": name, "multilineTextboxSpec": {"lines": [markdown]}},
        "position": {"x": x, "y": y, "width": w, "height": h},
    }


def bar(name, dataset, x_field, y_field, title, desc, pos,
        color_field=None, x_title=None, y_title=None) -> dict:
    fields = [x_field, y_field] + ([color_field] if color_field else [])
    encodings = {
        "x": {"fieldName": x_field, "scale": {"type": "categorical"},
              "displayName": x_title or x_field},
        "y": {"fieldName": y_field, "scale": {"type": "quantitative"},
              "displayName": y_title or y_field},
    }
    if color_field:
        encodings["color"] = {"fieldName": color_field,
                              "scale": {"type": "categorical"},
                              "displayName": color_field}
    px, py, pw, ph = pos
    return {
        "widget": {
            "name": name,
            "queries": _query(dataset, fields),
            "spec": {"version": 3, "widgetType": "bar",
                     "encodings": encodings, "frame": _frame(title, desc)},
        },
        "position": {"x": px, "y": py, "width": pw, "height": ph},
    }


def line(name, dataset, x_field, measures, title, desc, pos,
         x_title=None, y_title=None, x_scale="categorical") -> dict:
    """`measures` is [(field, display), ...] -- several lines on ONE scale.

    Deliberately one y-scale: a second axis would let two unrelated magnitudes
    share a picture and invent a correlation that is not in the data.
    """
    fields = [x_field] + [f for f, _ in measures]
    px, py, pw, ph = pos
    return {
        "widget": {
            "name": name,
            "queries": _query(dataset, fields),
            "spec": {
                "version": 3,
                "widgetType": "line",
                "encodings": {
                    "x": {"fieldName": x_field, "scale": {"type": x_scale},
                          "displayName": x_title or x_field},
                    "y": {"scale": {"type": "quantitative"},
                          "displayName": y_title or "value",
                          "fields": [{"fieldName": f, "displayName": d}
                                     for f, d in measures]},
                },
                "frame": _frame(title, desc),
            },
        },
        "position": {"x": px, "y": py, "width": pw, "height": ph},
    }


def table(name, dataset, columns, title, desc, pos) -> dict:
    """`columns` is [(field, display, type), ...] or [(field, display, type, rules), ...].

    `rules` is [(operator, value, hex), ...] -- a conditional background applied
    to that cell, first match winning. Only numeric comparisons are supported by
    the renderer, so the RAG colouring hangs off the PSI / delta columns rather
    than off the `status` strings beside them.
    """
    px, py, pw, ph = pos
    spec_columns = []
    for i, col in enumerate(columns):
        field, display, ctype = col[0], col[1], col[2]
        entry = {"fieldName": field, "displayName": display,
                 "type": ctype, "order": i, "visible": True}
        rules = col[3] if len(col) > 3 else None
        if rules:
            entry["style"] = {
                "type": "basic",
                "rules": [
                    {"condition": {"operand": {"type": "data-value", "value": str(value)},
                                   "operator": operator},
                     "backgroundColor": {"hex": hex_colour}}
                    for operator, value, hex_colour in rules
                ],
            }
        spec_columns.append(entry)
    return {
        "widget": {
            "name": name,
            "queries": _query(dataset, [c[0] for c in columns]),
            "spec": {
                "version": 2,
                "widgetType": "table",
                "encodings": {"columns": spec_columns},
                "frame": _frame(title, desc),
            },
        },
        "position": {"x": px, "y": py, "width": pw, "height": ph},
    }


def heatmap(name, dataset, x_field, y_field, value_field, aggregate,
            title, desc, pos, scheme="redyellowgreen") -> dict:
    """x categorical, y categorical, colour = an aggregate of `value_field`.

    Each (x, y) pair is already unique in these datasets, so the aggregate is
    an identity -- it is there because the colour encoding expects an aggregate
    expression, not because anything is being summarised.
    """
    px, py, pw, ph = pos
    measure = f"{aggregate}({value_field})"
    return {
        "widget": {
            "name": name,
            "queries": [{
                "name": "main_query",
                "query": {
                    "datasetName": dataset,
                    "fields": [
                        {"name": x_field, "expression": f"`{x_field}`"},
                        {"name": y_field, "expression": f"`{y_field}`"},
                        {"name": measure, "expression": f"{aggregate.upper()}(`{value_field}`)"},
                    ],
                    "disaggregated": False,
                },
            }],
            "spec": {
                "version": 3,
                "widgetType": "heatmap",
                "encodings": {
                    "x": {"fieldName": x_field, "scale": {"type": "categorical"},
                          "axis": {"hideTitle": True}},
                    "y": {"fieldName": y_field, "scale": {"type": "categorical"},
                          "axis": {"hideTitle": True}},
                    "color": {"fieldName": measure,
                              "scale": {"type": "quantitative",
                                        "colorRamp": {"mode": "scheme", "scheme": scheme}}},
                },
                "frame": _frame(title, desc),
            },
        },
        "position": {"x": px, "y": py, "width": pw, "height": ph},
    }


def counter(name, dataset, value_field, title, desc, pos, display=None) -> dict:
    """Single big number, read from the FIRST row of its dataset.

    `rowNumber` is 1-based and explicit: a counter silently showing row 1 of a
    multi-row result is a classic way to publish the wrong figure, so the
    datasets behind these tiles return exactly one row.
    """
    px, py, pw, ph = pos
    return {
        "widget": {
            "name": name,
            "queries": _query(dataset, [value_field]),
            "spec": {
                "version": 2,
                "widgetType": "counter",
                "encodings": {
                    "value": {"fieldName": value_field, "rowNumber": 1,
                              "displayName": display or value_field},
                },
                "frame": _frame(title, desc),
            },
        },
        "position": {"x": px, "y": py, "width": pw, "height": ph},
    }


def stacked_bar(name, dataset, x_field, measures, title, desc, pos,
                x_title=None, y_title=None, x_scale="categorical") -> dict:
    """Bar with SEVERAL measures on one scale, stacked.

    Separate from `bar()` on purpose: `bar()` encodes one measure plus an
    optional colour dimension, which is the right shape for a count per window.
    Two measures that sum to a meaningful total -- fraud caught plus fraud
    missed -- are a different picture and need `y.fields`.
    """
    px, py, pw, ph = pos
    return {
        "widget": {
            "name": name,
            "queries": _query(dataset, [x_field] + [f for f, _ in measures]),
            "spec": {
                "version": 3,
                "widgetType": "bar",
                "encodings": {
                    "x": {"fieldName": x_field, "scale": {"type": x_scale},
                          "displayName": x_title or x_field},
                    "y": {"scale": {"type": "quantitative"},
                          "displayName": y_title or "value",
                          "fields": [{"fieldName": f, "displayName": d}
                                     for f, d in measures]},
                },
                "frame": _frame(title, desc),
            },
        },
        "position": {"x": px, "y": py, "width": pw, "height": ph},
    }


INTRO = (
    "# Fraud model drift watch\n\n"
    "Equal-width `step` windows over the post-training period, each compared "
    "against the training reference. The tiles keep three different questions "
    "apart — **did the inputs move** (Q3/Q3b), **did the scores move** (Q2/Q4), "
    "and **did the model actually get worse** (Q5) — because only the third one "
    "justifies a retrain.\n\n"
    "Every query reads the small aggregate tables written by `06_generate_monitoring`, "
    "never the 6.3 M-row Gold table."
)

READING_NOTE = (
    "### Reading the rollup\n"
    "`overall_status` is worst-of the three families. **Performance reads "
    "`UNKNOWN` on the most recent windows by design** — fraud labels arrive with "
    "chargebacks and investigations, days to weeks late. A missing window would "
    "read as \"fine\"; `UNKNOWN` reads as \"not yet knowable\".\n\n"
    "Drift red + performance unknown is *investigate*, not *retrain*."
)


# PSI banding, reused by the conditional cell colours. Same numbers the SQL
# file documents: < 0.10 stable, 0.10-0.25 moderate, > 0.25 significant.
RED, AMBER, GREEN = "#d13212", "#e8a33d", "#1d8102"
PSI_RULES = [(">=", 0.25, RED), (">=", 0.10, AMBER), (">=", 0.0, GREEN)]


def build_layout() -> list[dict]:
    y = 0
    layout = [text("intro", INTRO, (0, y, 12, 3))]
    y += 3

    layout.append(table(
        "q6_rollup", "q6_rag_indicator",
        [("window_id", "Window", "string"),
         ("overall_status", "Overall", "string"),
         ("feature_drift_status", "Features", "string"),
         ("prediction_drift_status", "Scores", "string"),
         ("performance_status", "Performance", "string"),
         ("worst_feature_psi", "Worst feature PSI", "float", PSI_RULES),
         ("prediction_score_psi", "Score PSI", "float", PSI_RULES),
         ("pr_auc", "PR-AUC", "float"),
         ("pr_auc_delta", "Δ PR-AUC", "float"),
         ("row_count", "Rows", "integer"),
         ("interpretation", "Interpretation", "string")],
        "Window verdicts",
        "Worst-of the three families, each family's own verdict kept visible.",
        (0, y, 12, 8)))
    y += 8
    layout.append(text("rollup_note", READING_NOTE, (0, y, 12, 3)))
    y += 3

    layout.append(bar(
        "q1_volume_bar", "q1_volume", "window_id", "transaction_volume",
        "Transactions per window",
        "The denominator for every other tile. A volume collapse explains an "
        "alert-rate change far more often than model decay does.",
        (0, y, 6, 6), x_title="Window", y_title="Transactions"))
    layout.append(line(
        "q4_alert_rate", "q4_prediction_drift", "window_id",
        [("predicted_positive_rate_pct", "Window alert rate %"),
         ("reference_positive_rate_pct", "Training reference %")],
        "Alert rate vs. training reference",
        "Share of transactions at or above the operational threshold. A staffing "
        "question before it is a modelling question.",
        (6, y, 6, 6), x_title="Window", y_title="% of transactions"))
    y += 6

    layout.append(line(
        "q5_perf", "q5_performance", "window_id",
        [("precision_at_threshold", "Precision"),
         ("recall_at_threshold", "Recall"),
         ("pr_auc", "PR-AUC")],
        "Precision, recall, PR-AUC",
        "The only tile that can prove degradation — and it stops where the labels "
        "stop. Recent windows are intentionally blank.",
        (0, y, 6, 6), x_title="Window", y_title="Score"))
    layout.append(heatmap(
        "q2_score_bands", "q2_score_histogram", "window_id", "score_band",
        "row_share", "sum",
        "Where the scores sit",
        "Share of each window's rows per score band. Mass can migrate to the tails "
        "while the mean holds steady.",
        (6, y, 6, 6), scheme="blues"))
    y += 6

    layout.append(heatmap(
        "q3_psi", "q3_feature_psi", "window_id", "feature", "psi", "avg",
        "Feature drift (PSI) by feature and window",
        "Which input moved, and when. Under 0.10 stable · 0.10–0.25 moderate · "
        "above 0.25 significant.",
        (0, y, 12, 8), scheme="redyellowgreen"))
    y += 8

    layout.append(table(
        "q3b_missing", "q3b_missing_rate",
        [("feature", "Feature", "string"),
         ("window_id", "Window", "string"),
         ("missing_rate_delta_vs_reference", "Δ missing rate", "float"),
         ("status", "Status", "string")],
        "Missing-rate change vs. reference",
        "Nulls in a served feature are a plumbing failure and move independently "
        "of PSI. An empty table is the healthy state.",
        (0, y, 6, 6)))
    layout.append(table(
        "q7_quality", "q7_data_quality",
        [("rule_name", "Rule", "string"),
         ("severity", "Severity", "string"),
         ("failed_row_count", "Failed", "integer"),
         ("evaluated_row_count", "Evaluated", "integer"),
         ("failure_rate_pct", "Failure %", "float"),
         ("description", "Why it exists", "string")],
        "Bronze → Silver data quality",
        "Proves the ingest contract held for this run. Reviewers ask for this "
        "before they trust any metric above it.",
        (6, y, 6, 6)))
    y += 6

    layout.append(table(
        "q8_contract", "q8_feature_contract",
        [("signature_position", "#", "integer"),
         ("feature_name", "Column", "string"),
         ("dtype", "Type", "string"),
         ("availability", "Availability", "string"),
         ("served_to_endpoint", "Served", "boolean"),
         ("derivation", "Derivation", "string"),
         ("description", "Description", "string")],
        "Model and feature contract",
        "What the endpoint may see, in signature order — and which Gold columns "
        "are withheld, so the leakage decisions stay auditable.",
        (0, y, 12, 8)))
    return layout


IMPACT_INTRO = (
    "# Fraud loss prevention — simulated business impact\n\n"
    "What the model would be **worth** if it were used to block or review "
    "suspicious transactions, swept across every decision threshold from 0.05 "
    "to 0.95.\n\n"
    "> ⚠️ **The savings shown are simulated estimates based on historical "
    "transaction labels, model predictions, and configurable business "
    "assumptions. They should not be interpreted as guaranteed real-world "
    "savings without production validation.**\n\n"
    "Read the assumptions tile *before* the number beside it: three of the four "
    "inputs to every figure on this page are assumed, not measured."
)

IMPACT_METHOD_NOTE = (
    "### How a saving is computed\n"
    "`gross = detected_fraud_amount × prevention_rate` · "
    "`review_cost = flagged × cost_per_review` · "
    "`friction = false_positives × customer_friction_cost` · "
    "**`net = gross − review_cost − friction`**\n\n"
    "Reviews are charged on false positives too — the queue cannot tell them "
    "apart until they are worked. Not modelled at all: recovery on missed "
    "fraud, the time-value of a blocked payment, churn, and the cost of running "
    "the platform. Treating those as zero is more honest than inventing a "
    "number for them."
)

IMPACT_CLOSING_NOTE = (
    "### Before quoting any figure above\n"
    "The recommended threshold is simply the row with the highest estimated net "
    "savings — a **starting point for a conversation, not a decision**. A real "
    "operating point also has to fit the review capacity that actually exists "
    "(tile D) and the organisation's tolerance for stopping good customers.\n\n"
    "PaySim is a *synthetic* dataset, so the shape of the trade-off is the "
    "transferable finding; the dollar amounts are illustrative only."
)


# The counter is deliberately narrower than the grid: it is a hero number, and
# the two tables beneath it were widened to the full 12 columns in the workspace
# because a long free-text column squeezed into a third of the page is
# unreadable. That rearrangement was adopted here rather than overwritten, so it
# survives every future deploy. `tests/test_dashboard_spec.py` records the
# resulting short row as the one intentional exception to the fill-the-grid rule.
HERO_COUNTER_ROW = "a1_net_savings_counter"


def build_impact_layout() -> list[dict]:
    y = 0
    layout = [text("impact_intro", IMPACT_INTRO, (0, y, 12, 4))]
    y += 4

    # The headline, then the assumptions that produced it -- both above any
    # chart, so nobody reaches the analysis without having seen the inputs.
    layout.append(counter(
        "a1_net_savings_counter", "a1_net_savings", "estimated_net_savings",
        "Estimated NET savings (simulated)",
        "Fraud value prevented, minus the cost of reviewing the queue, minus the "
        "cost of stopping good customers. The only figure here a finance "
        "function would recognise as a benefit.",
        (0, y, 4, 5), display="Estimated net savings"))
    y += 5
    layout.append(table(
        "a2_kpi_table", "a2_kpi_tiles",
        [("metric", "Metric", "string"),
         ("value", "Value", "float"),
         ("unit", "Unit", "string"),
         ("currency", "Currency", "string")],
        "Executive summary",
        "Benefit, then gross, then what was caught — then what was NOT. A page "
        "that stops after the first two rows is a sales pitch, not an analysis.",
        (0, y, 12, 5)))
    y += 5
    layout.append(table(
        "a3_assumptions_table", "a3_assumptions",
        [("assumption", "Assumption", "string"),
         ("value", "Value", "string")],
        "Assumptions in force",
        "If one of these looks wrong, stop and fix it before discussing the "
        "number above it. Change them in config.py and re-run notebook 08.",
        (0, y, 12, 5)))
    y += 5

    # The analytical centre, full width: this is the chart that explains why the
    # recommendation sits where it does.
    layout.append(line(
        "b1_tradeoff_line", "b1_threshold_tradeoff", "threshold",
        [("estimated_net_savings", "Net savings"),
         ("gross_estimated_savings", "Gross fraud prevented"),
         ("review_cost", "Review cost"),
         ("false_positive_friction_cost", "False-positive friction cost")],
        "Savings and costs across the threshold sweep",
        "Gross savings fall as the threshold rises (fewer frauds caught) while "
        "both costs fall faster at the low end — so net savings peaks in "
        "between. A flat peak means the exact threshold barely matters.",
        (0, y, 12, 7), x_title="Decision threshold", y_title="Amount",
        x_scale="quantitative"))
    y += 7
    layout.append(text("impact_method", IMPACT_METHOD_NOTE, (0, y, 12, 3)))
    y += 3

    layout.append(stacked_bar(
        "c1_caught_vs_missed_bar", "c1_caught_vs_missed", "threshold",
        [("detected_fraud_amount", "Fraud value caught"),
         ("missed_fraud_amount", "Fraud value missed")],
        "Fraud value caught vs let through",
        "Recall, in money. The two always sum to the total fraudulent value. "
        "80 % recall sounds strong until the missing 20 % is the large-value "
        "transactions — which a recall number hides and this chart does not.",
        (0, y, 6, 6), x_title="Decision threshold", y_title="Fraud value"))
    layout.append(bar(
        "d1_workload_bar", "d1_review_workload", "threshold",
        "flagged_transaction_count",
        "Review queue by threshold",
        "The constraint that usually decides the operating point in practice. "
        "See the days-required tile below before adopting a threshold.",
        (6, y, 6, 6), x_title="Decision threshold", y_title="Transactions flagged"))
    y += 6

    layout.append(line(
        "d1_review_days", "d1_review_workload", "threshold",
        [("estimated_review_days_required", "Analyst-days required")],
        "Analyst-days implied by the alert volume",
        "Alert volume divided by one analyst's daily capacity. A threshold that "
        "maximises modelled savings but needs forty analyst-days per batch is "
        "not an option, however good it looks above.",
        (0, y, 6, 6), x_title="Decision threshold", y_title="Analyst-days",
        x_scale="quantitative"))
    layout.append(line(
        "e1_quality_line", "e1_quality_vs_value", "threshold",
        [("precision_pct", "Precision %"),
         ("recall_pct", "Recall %"),
         ("false_positive_rate_pct", "False-positive rate %")],
        "Model quality across the same sweep",
        "The bridge between the two dashboards. The threshold that maximises "
        "money is rarely the one that maximises F1: a false positive and a true "
        "positive are not worth the same amount.",
        (6, y, 6, 6), x_title="Decision threshold", y_title="Percent",
        x_scale="quantitative"))
    y += 6

    layout.append(table(
        "f2_neighbouring_table", "f2_neighbouring",
        [("threshold", "Threshold", "float"),
         ("is_recommended", "Recommended", "boolean"),
         ("estimated_net_savings", "Net savings", "float"),
         ("pct_of_best_net_savings", "% of best", "float"),
         ("flagged_transaction_count", "Flagged", "integer"),
         ("estimated_review_days_required", "Analyst-days", "float"),
         ("precision_pct", "Precision %", "float"),
         ("recall_pct", "Recall %", "float")],
        "Neighbouring thresholds — how much does the choice matter?",
        "Guards against false precision. If two thresholds are worth within a "
        "percent of each other, pick on review capacity, not on the optimum.",
        (0, y, 12, 6)))
    y += 6

    layout.append(table(
        "f1_recommendation_table", "f1_recommendation",
        [("recommended_threshold", "Threshold", "float"),
         ("recommendation_note", "Recommendation", "string"),
         ("selection_basis", "Chosen by", "string"),
         ("business_override_note", "Why you might choose otherwise", "string"),
         ("result_disclaimer", "Disclaimer", "string"),
         ("simulated_at", "Simulated at", "string")],
        "Business recommendation",
        "The conclusion, its basis, and the validation caveat in one place — so "
        "the page cannot be screenshotted into a slide without them.",
        (0, y, 12, 4)))
    y += 4
    layout.append(text("impact_closing", IMPACT_CLOSING_NOTE, (0, y, 12, 3)))
    return layout


def build_impact(queries: dict[str, str]) -> dict:
    return {
        "datasets": [
            {"name": IMPACT_DATASETS[qid][0],
             "displayName": IMPACT_DATASETS[qid][1],
             "queryLines": [l + "\n" for l in queries[qid].splitlines()]}
            for qid in IMPACT_DATASETS
        ],
        "pages": [{
            "name": "business_impact",
            "displayName": "Business impact",
            "layout": build_impact_layout(),
        }],
    }


OPS_INTRO = (
    "# Fraud detection in operation — a replay\n\n"
    "The model at **one frozen threshold**, hour by hour, as if it had been "
    "shipped: how many frauds it caught, what that saved, and what the alert "
    "queue cost to work.\n\n"
    "> ⚠️ **Two caveats that change how every tile is read.** The dates are "
    "**synthetic** — PaySim ships only an hour counter, anchored to 2024-01-01 "
    "so a chart can have a time axis. And this is a **replay, not a production "
    "record**: the model is applied to transactions it did not influence, so "
    "no fraudster ever adapted to it and no analyst ever worked a queue.\n\n"
    "> The savings are simulated estimates under configurable business "
    "assumptions. They should not be interpreted as guaranteed real-world "
    "savings without production validation."
)

OPS_MONEY_NOTE = (
    "### Why cost is smooth and savings are spiky\n"
    "Operational cost tracks alert **volume** — reviews and false-positive "
    "friction — so it moves with traffic. Gross savings tracks transaction "
    "**value**, and one large fraudulent transfer moves it more than a busy "
    "hour of small ones. A spiky benefit line against a flat cost line is the "
    "expected shape, not an anomaly.\n\n"
    "`operational_cost = review_cost + false_positive_friction_cost` — the same "
    "two components the business impact dashboard subtracts. One definition of "
    "cost across both pages."
)

OPS_CAPACITY_NOTE = (
    "### The constraint that model metrics cannot see\n"
    "Alerts above the capacity line do not disappear — they become a backlog. "
    "A fraud desk usually fails by being **behind**, not by being wrong, and no "
    "amount of PR-AUC answers whether a night shift is needed. Capacity here is "
    "`analysts × daily capacity`, scaled to the period grain."
)


def build_ops_layout() -> list[dict]:
    y = 0
    layout = [text("ops_intro", OPS_INTRO, (0, y, 12, 5))]
    y += 5

    layout.append(counter(
        "t2_ratio_counter", "t2_savings_ratio", "savings_per_unit_cost",
        "Gross savings per 1 unit spent reviewing",
        "Unitless, so it survives a change of currency and compares two "
        "operating points fairly. The most quotable number on the page.",
        (0, y, 4, 5), display="Return per unit of review spend"))
    layout.append(table(
        "t1_golive_table", "t1_since_golive",
        [("periods_observed", "Periods", "integer"),
         ("operating_threshold", "Threshold", "float"),
         ("total_alerts", "Alerts", "integer"),
         ("total_caught_fraud_count", "Fraud caught", "integer"),
         ("total_missed_fraud_count", "Fraud missed", "integer"),
         ("gross_savings", "Gross savings", "float"),
         ("operational_cost", "Operational cost", "float"),
         ("net_savings", "Net savings", "float"),
         ("periods_over_capacity", "Periods over capacity", "integer"),
         ("result_disclaimer", "Disclaimer", "string")],
        "Since go-live",
        "The operational bottom line for the whole replayed period. Everything "
        "below is this row decomposed over time.",
        (4, y, 8, 5)))
    y += 5

    layout.append(line(
        "u2_payback_curve", "u2_cumulative", "period_start",
        [("cumulative_net_savings", "Cumulative net savings"),
         ("cumulative_gross_savings", "Cumulative gross savings"),
         ("cumulative_operational_cost", "Cumulative operational cost")],
        "The payback curve",
        "A rising line means the deployment is compounding; a flat stretch means "
        "the model found nothing while the desk kept being paid. If cost ever "
        "crosses above savings, the deployment is under water.",
        (0, y, 12, 7), x_title="Time", y_title="Cumulative amount",
        x_scale="temporal"))
    y += 7

    layout.append(line(
        "u1_savings_vs_cost", "u1_savings_vs_cost", "period_start",
        [("gross_savings", "Gross savings"),
         ("operational_cost", "Operational cost"),
         ("net_savings", "Net savings")],
        "Savings vs operational cost, period by period",
        "What the model prevented, and what preventing it cost to operate, on "
        "the same clock. The stability of the gap matters more than its size in "
        "any one period.",
        (0, y, 12, 7), x_title="Time", y_title="Amount", x_scale="temporal"))
    y += 7
    layout.append(text("ops_money_note", OPS_MONEY_NOTE, (0, y, 12, 3)))
    y += 3

    layout.append(stacked_bar(
        "u3_daily_bar", "u3_daily_rollup", "event_date",
        [("operational_cost", "Operational cost"),
         ("net_savings", "Net savings")],
        "Daily roll-up — cost and benefit",
        "The two segments sum to gross savings. Same data as the hourly series, "
        "aggregated — the version that goes on a slide without an argument "
        "about noise.",
        (0, y, 6, 6), x_title="Date", y_title="Amount", x_scale="temporal"))
    layout.append(line(
        "w1_capacity", "w1_alerts_vs_capacity", "period_start",
        [("alerts", "Alerts raised"),
         ("review_capacity", "Staffed review capacity"),
         ("backlog", "Backlog")],
        "Alert volume against staffed capacity",
        "Alerts above the capacity line become a backlog. This tile decides "
        "headcount, and no model metric can answer it.",
        (6, y, 6, 6), x_title="Time", y_title="Alerts", x_scale="temporal"))
    y += 6

    layout.append(stacked_bar(
        "v1_counts", "v1_caught_vs_missed", "period_start",
        [("fraud_caught", "Fraud caught"),
         ("fraud_missed", "Fraud missed")],
        "Fraud caught vs missed — count",
        "Recall as a running record. A recall figure is a number; a bar that "
        "turns in one specific hour is an incident someone can investigate.",
        (0, y, 6, 6), x_title="Time", y_title="Transactions", x_scale="temporal"))
    layout.append(stacked_bar(
        "v2_value", "v2_value_caught_vs_missed", "period_start",
        [("fraud_value_caught", "Fraud value caught"),
         ("fraud_value_missed", "Fraud value missed")],
        "Fraud caught vs missed — value",
        "The same picture weighted by money. A period where the counts look "
        "fine and this does not is a period that missed the expensive ones.",
        (6, y, 6, 6), x_title="Time", y_title="Fraud value", x_scale="temporal"))
    y += 6

    layout.append(line(
        "w2_hour_profile", "w2_hour_profile", "event_hour_of_day",
        [("avg_alerts", "Average alerts"),
         ("review_capacity", "Staffed capacity"),
         ("peak_alerts", "Peak alerts")],
        "Alert load by hour of day",
        "Shift planning. Fraud is not uniform across the clock, so neither is "
        "the queue — this is the tile that answers \"do we need a night shift?\".",
        (0, y, 6, 6), x_title="Hour of day", y_title="Alerts"))
    layout.append(line(
        "x1_quality", "x1_quality_over_time", "period_start",
        [("precision_pct", "Precision %"),
         ("recall_pct", "Recall %")],
        "Detection quality over time",
        "Noisy on low-volume periods by construction: a period with two frauds "
        "has a recall of 0, 50 or 100 % and nothing in between. Periods with no "
        "fraud at all are excluded rather than plotted as zero.",
        (6, y, 6, 6), x_title="Time", y_title="Percent", x_scale="temporal"))
    y += 6
    layout.append(text("ops_capacity_note", OPS_CAPACITY_NOTE, (0, y, 12, 3)))
    y += 3

    layout.append(table(
        "x2_worst_periods", "x2_worst_periods",
        [("period_start", "Period", "datetime"),
         ("issue", "Issue", "string"),
         ("alerts", "Alerts", "integer"),
         ("fraud_caught", "Caught", "integer"),
         ("fraud_missed", "Missed", "integer"),
         ("fraud_value_missed", "Value missed", "float"),
         ("operational_cost", "Operational cost", "float"),
         ("net_savings", "Net savings", "float"),
         ("backlog", "Backlog", "float")],
        "Worst operating periods",
        "The exceptions view — every period that lost money, missed fraud, or "
        "overflowed the queue. An empty table is the healthy state.",
        (0, y, 12, 7)))
    return layout


def build_ops(queries: dict[str, str]) -> dict:
    return {
        "datasets": [
            {"name": OPS_DATASETS[qid][0],
             "displayName": OPS_DATASETS[qid][1],
             "queryLines": [l + "\n" for l in queries[qid].splitlines()]}
            for qid in OPS_DATASETS
        ],
        "pages": [{
            "name": "operational_replay",
            "displayName": "Operational replay",
            "layout": build_ops_layout(),
        }],
    }


def build(queries: dict[str, str]) -> dict:
    return {
        "datasets": [
            {"name": DATASETS[qid][0],
             "displayName": DATASETS[qid][1],
             "queryLines": [l + "\n" for l in queries[qid].splitlines()]}
            for qid in DATASETS
        ],
        "pages": [{
            "name": "drift_watch",
            "displayName": "Drift watch",
            "layout": build_layout(),
        }],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if the committed JSON is stale")
    args = ap.parse_args()

    targets = [
        (OUT_PATH, build(parse_queries(SQL_PATH.read_text()))),
        (IMPACT_OUT_PATH, build_impact(parse_queries(
            IMPACT_SQL_PATH.read_text(), IMPACT_DATASETS, IMPACT_HEADER_RE,
            IMPACT_SQL_PATH))),
        (OPS_OUT_PATH, build_ops(parse_queries(
            OPS_SQL_PATH.read_text(), OPS_DATASETS, OPS_HEADER_RE,
            OPS_SQL_PATH))),
    ]

    stale = 0
    for path, doc in targets:
        rendered = json.dumps(doc, indent=2, ensure_ascii=False) + "\n"
        if args.check:
            current = path.read_text() if path.exists() else ""
            if current != rendered:
                print(f"{path.name} is out of date -- rerun build_lvdash.py",
                      file=sys.stderr)
                stale += 1
            else:
                print(f"{path.name} is up to date")
            continue
        path.write_text(rendered)
        print(f"wrote {path} "
              f"({len(doc['datasets'])} datasets, "
              f"{len(doc['pages'][0]['layout'])} widgets)")
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
