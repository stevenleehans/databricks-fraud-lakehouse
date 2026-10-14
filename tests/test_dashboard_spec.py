"""Static checks on the generated Lakeview dashboard.

The dashboard is the artefact a reviewer actually looks at, and every failure
mode here is silent in the UI: a mistyped `fieldName` renders an empty tile, a
stale JSON shows last week's thresholds, and a tile pointed at the Gold table
stalls a Free Edition warehouse instead of erroring. None of that surfaces until
someone is demoing.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
DASH_DIR = REPO / "dashboards"
JSON_PATH = DASH_DIR / "fraud_drift_monitoring.lvdash.json"
IMPACT_JSON_PATH = DASH_DIR / "fraud_business_impact.lvdash.json"
OPS_JSON_PATH = DASH_DIR / "fraud_operations_replay.lvdash.json"
GENERATOR = DASH_DIR / "build_lvdash.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("build_lvdash", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ALL_DASHBOARDS = {
    "drift": JSON_PATH,
    "impact": IMPACT_JSON_PATH,
    "operations": OPS_JSON_PATH,
}


@pytest.fixture(scope="module", params=sorted(ALL_DASHBOARDS))
def dashboard(request) -> dict:
    """Every dashboard. The rules asserted through this fixture are properties
    of the Lakeview renderer -- grid arithmetic, spec versions, field wiring --
    and apply equally to whichever page is rendering."""
    return json.loads(ALL_DASHBOARDS[request.param].read_text())


@pytest.fixture(scope="module")
def drift_dashboard() -> dict:
    return json.loads(JSON_PATH.read_text())


@pytest.fixture(scope="module")
def impact_dashboard() -> dict:
    return json.loads(IMPACT_JSON_PATH.read_text())


@pytest.fixture(scope="module")
def ops_dashboard() -> dict:
    return json.loads(OPS_JSON_PATH.read_text())


@pytest.fixture(scope="module")
def generator():
    return _load_generator()


def _query_text(dataset: dict) -> str:
    return "".join(dataset["queryLines"])


def _widgets(dashboard: dict):
    for page in dashboard["pages"]:
        for entry in page["layout"]:
            yield entry


def test_the_committed_json_matches_the_generator(generator, drift_dashboard):
    """The SQL file is the source of truth; the JSON is derived from it.

    If this fails, someone edited one of the two without rerunning
    `python dashboards/build_lvdash.py`, and the dashboard now disagrees with
    the reviewed queries.
    """
    regenerated = generator.build(generator.parse_queries(generator.SQL_PATH.read_text()))
    assert regenerated == drift_dashboard, "run dashboards/build_lvdash.py"


def test_the_committed_impact_json_matches_the_generator(generator, impact_dashboard):
    regenerated = generator.build_impact(generator.parse_queries(
        generator.IMPACT_SQL_PATH.read_text(),
        generator.IMPACT_DATASETS,
        generator.IMPACT_HEADER_RE,
        generator.IMPACT_SQL_PATH,
    ))
    assert regenerated == impact_dashboard, "run dashboards/build_lvdash.py"


def test_every_sql_query_became_a_dataset(generator, drift_dashboard):
    expected = {name for name, _ in generator.DATASETS.values()}
    assert {d["name"] for d in drift_dashboard["datasets"]} == expected


def test_every_impact_sql_query_became_a_dataset(generator, impact_dashboard):
    expected = {name for name, _ in generator.IMPACT_DATASETS.values()}
    assert {d["name"] for d in impact_dashboard["datasets"]} == expected


def test_the_committed_ops_json_matches_the_generator(generator, ops_dashboard):
    regenerated = generator.build_ops(generator.parse_queries(
        generator.OPS_SQL_PATH.read_text(),
        generator.OPS_DATASETS,
        generator.OPS_HEADER_RE,
        generator.OPS_SQL_PATH,
    ))
    assert regenerated == ops_dashboard, "run dashboards/build_lvdash.py"


def test_every_ops_sql_query_became_a_dataset(generator, ops_dashboard):
    expected = {name for name, _ in generator.OPS_DATASETS.values()}
    assert {d["name"] for d in ops_dashboard["datasets"]} == expected


def test_queries_carry_no_trailing_semicolon(dashboard):
    """A Lakeview dataset holds one statement; a trailing `;` is rejected."""
    for ds in dashboard["datasets"]:
        assert not _query_text(ds).strip().endswith(";"), ds["name"]


def test_no_tile_reads_the_gold_table(dashboard):
    """Free Edition budget rule: dashboards read the small monitoring
    aggregates, never the 6.3 M-row Gold table."""
    for ds in dashboard["datasets"]:
        assert "gold_transaction_features" not in _query_text(ds), ds["name"]


def test_every_widget_points_at_a_dataset_that_exists(dashboard):
    names = {d["name"] for d in dashboard["datasets"]}
    for entry in _widgets(dashboard):
        for query in entry["widget"].get("queries", []):
            assert query["query"]["datasetName"] in names, entry["widget"]["name"]


def _encoded_field_names(spec: dict):
    for encoding in spec.get("encodings", {}).values():
        if isinstance(encoding, dict):
            if "fieldName" in encoding:
                yield encoding["fieldName"]
            for field in encoding.get("fields", []):
                yield field["fieldName"]
        elif isinstance(encoding, list):          # table columns
            for column in encoding:
                yield column["fieldName"]


def test_every_encoded_field_is_selected_by_its_query(dashboard):
    """A `fieldName` the query does not return renders as an empty tile rather
    than an error, so it has to be caught here."""
    by_name = {d["name"]: _query_text(d) for d in dashboard["datasets"]}
    for entry in _widgets(dashboard):
        widget = entry["widget"]
        spec = widget.get("spec")
        if not spec:
            continue                              # text widget
        dataset = widget["queries"][0]["query"]["datasetName"]
        sql = by_name[dataset]
        selected = {f["name"] for f in widget["queries"][0]["query"]["fields"]}
        for field in _encoded_field_names(spec):
            assert field in selected, f"{widget['name']}: {field} not requested"
            # a colour encoding may be an aggregate, e.g. `avg(psi)`
            bare = re.sub(r"^\w+\((.*)\)$", r"\1", field)
            assert re.search(rf"\b{re.escape(bare)}\b", sql), \
                f"{widget['name']}: {bare} is not produced by {dataset}"


def test_tiles_do_not_overlap(dashboard):
    """Lakeview will happily render two widgets on top of each other."""
    for page in dashboard["pages"]:
        boxes = [(e["position"], e["widget"]["name"]) for e in page["layout"]]
        for i, (a, name_a) in enumerate(boxes):
            assert a["x"] + a["width"] <= 12, f"{name_a} runs off the 12-column grid"
            for b, name_b in boxes[i + 1:]:
                overlap = (a["x"] < b["x"] + b["width"] and b["x"] < a["x"] + a["width"]
                           and a["y"] < b["y"] + b["height"] and b["y"] < a["y"] + a["height"])
                assert not overlap, f"{name_a} overlaps {name_b}"


#: Rows that are deliberately narrower than the grid. A hero counter reads
#: better at a third of the page than stretched across it, and the impact
#: dashboard's arrangement was adopted from the workspace rather than
#: overwritten. Anything not listed here must still fill the 12 columns.
INTENTIONALLY_SHORT_ROWS = {"a1_net_savings_counter"}


def test_every_row_fills_the_grid(dashboard):
    """Lakeview lays out on 12 columns. A row summing to less than 12 leaves a
    ragged gap that reads as a missing tile -- unless it is one of the few rows
    that is short on purpose."""
    for page in dashboard["pages"]:
        rows: dict[int, int] = {}
        occupants: dict[int, set] = {}
        for entry in page["layout"]:
            pos = entry["position"]
            rows[pos["y"]] = rows.get(pos["y"], 0) + pos["width"]
            occupants.setdefault(pos["y"], set()).add(entry["widget"]["name"])
        for y, width in rows.items():
            if occupants[y] & INTENTIONALLY_SHORT_ROWS:
                continue
            assert width == 12, f"row y={y} sums to {width}, not 12"


def test_widget_spec_versions_match_the_renderer(dashboard):
    """The version is what the *renderer* dispatches on. A table at version 1
    round-trips through the API perfectly and then renders as
    'Visualization has no fields selected' -- so it has to be asserted here."""
    expected = {"table": 2, "bar": 3, "line": 3, "heatmap": 3, "counter": 2}
    for entry in _widgets(dashboard):
        spec = entry["widget"].get("spec")
        if not spec:
            continue
        widget_type = spec["widgetType"]
        assert widget_type in expected, f"unreviewed widget type {widget_type}"
        assert spec["version"] == expected[widget_type], \
            f"{entry['widget']['name']}: {widget_type} must be version {expected[widget_type]}"


def test_text_widgets_use_the_canonical_shape(dashboard):
    """`textbox_spec` is silently ignored; the renderer reads
    `multilineTextboxSpec.lines`. Entries in that array are concatenated onto
    one line, so the markdown must arrive as a single entry."""
    for entry in _widgets(dashboard):
        widget = entry["widget"]
        if "spec" in widget:
            continue
        assert "multilineTextboxSpec" in widget, widget["name"]
        assert len(widget["multilineTextboxSpec"]["lines"]) == 1, widget["name"]


def test_the_rollup_tile_leads_the_page(drift_dashboard):
    """Q6 is the headline. If it slips below a chart, the page starts with a
    drift number instead of the verdict that says drift is not degradation."""
    layout = drift_dashboard["pages"][0]["layout"]
    charts = [e for e in layout if e["widget"].get("spec")]
    assert charts[0]["widget"]["name"] == "q6_rollup"


# --------------------------------------------------------------------------- #
# Business impact dashboard
#
# The failure mode guarded here is presentational, not technical: a savings
# figure that reaches a slide without the assumptions and the caveat that make
# it an estimate. None of that shows up as an error in the UI.
# --------------------------------------------------------------------------- #

def test_the_net_savings_counter_leads_the_impact_page(impact_dashboard):
    layout = impact_dashboard["pages"][0]["layout"]
    charts = [e for e in layout if e["widget"].get("spec")]
    assert charts[0]["widget"]["name"] == "a1_net_savings_counter"


def test_the_assumptions_tile_comes_before_any_chart(impact_dashboard):
    """The assumptions need not sit beside the headline -- they were widened to
    the full page in the workspace -- but nobody may reach the analysis without
    having passed them."""
    by_name = {e["widget"]["name"]: e["position"]
               for e in impact_dashboard["pages"][0]["layout"]}
    assumptions_y = by_name["a3_assumptions_table"]["y"]
    assert assumptions_y > by_name["a1_net_savings_counter"]["y"]
    assert assumptions_y < by_name["b1_tradeoff_line"]["y"]


def test_the_impact_page_carries_the_simulation_disclaimer(impact_dashboard):
    text_blocks = " ".join(
        " ".join(e["widget"]["multilineTextboxSpec"]["lines"])
        for e in impact_dashboard["pages"][0]["layout"]
        if "multilineTextboxSpec" in e["widget"]
    )
    assert "simulated estimates" in text_blocks
    assert "production validation" in text_blocks


def test_the_recommendation_tile_shows_its_own_disclaimer_column(impact_dashboard):
    """A screenshot of the recommendation must carry the caveat with it."""
    fields = set()
    for entry in impact_dashboard["pages"][0]["layout"]:
        widget = entry["widget"]
        if widget["name"] == "f1_recommendation_table":
            fields = {f["name"] for f in widget["queries"][0]["query"]["fields"]}
    assert "result_disclaimer" in fields
    assert "business_override_note" in fields


def test_no_impact_tile_reads_the_row_level_scored_table(impact_dashboard):
    """Same Free Edition budget rule as the drift page: dashboards read the two
    tiny Gold aggregates, never the millions of scored rows behind them."""
    for ds in impact_dashboard["datasets"]:
        assert "scored_predictions" not in _query_text(ds), ds["name"]
        assert "gold_transaction_features" not in _query_text(ds), ds["name"]


# --------------------------------------------------------------------------- #
# Operational replay dashboard
#
# This is the only page whose x-axis is clock time, and its two honesty risks
# are specific to that: dates that look real but are not, and a replay that
# reads as a production record.
# --------------------------------------------------------------------------- #

def test_the_ops_page_says_the_dates_are_synthetic(ops_dashboard):
    text_blocks = " ".join(
        " ".join(e["widget"]["multilineTextboxSpec"]["lines"])
        for e in ops_dashboard["pages"][0]["layout"]
        if "multilineTextboxSpec" in e["widget"]
    ).lower()
    assert "synthetic" in text_blocks
    assert "replay, not a production record" in text_blocks
    assert "production validation" in text_blocks


def test_the_ops_page_leads_with_the_return_ratio(ops_dashboard):
    """The ratio needs no currency and no context to be understood, so it opens
    the page ahead of any absolute figure."""
    charts = [e for e in ops_dashboard["pages"][0]["layout"] if e["widget"].get("spec")]
    assert charts[0]["widget"]["name"] == "t2_ratio_counter"


def test_the_ops_page_plots_savings_against_operational_cost(ops_dashboard):
    """The whole point of the page: the two series on one clock."""
    fields = set()
    for entry in ops_dashboard["pages"][0]["layout"]:
        if entry["widget"]["name"] == "u1_savings_vs_cost":
            fields = {f["name"] for f in entry["widget"]["queries"][0]["query"]["fields"]}
    assert {"gross_savings", "operational_cost", "net_savings"} <= fields


def test_the_ops_page_carries_the_capacity_view(ops_dashboard):
    """Alert volume without a capacity line is a chart nobody can act on."""
    names = {e["widget"]["name"] for e in ops_dashboard["pages"][0]["layout"]}
    assert "w1_capacity" in names
    assert "w2_hour_profile" in names


def test_time_series_tiles_use_a_temporal_x_axis(ops_dashboard):
    """A time axis rendered as 200-odd categorical labels is unreadable, and
    the failure is silent -- the tile still renders."""
    for entry in ops_dashboard["pages"][0]["layout"]:
        spec = entry["widget"].get("spec")
        if not spec:
            continue
        x = spec.get("encodings", {}).get("x", {})
        if x.get("fieldName") == "period_start":
            assert x["scale"]["type"] == "temporal", entry["widget"]["name"]


def test_no_ops_tile_reads_the_row_level_scored_table(ops_dashboard):
    for ds in ops_dashboard["datasets"]:
        assert "scored_predictions" not in _query_text(ds), ds["name"]
        assert "gold_transaction_features" not in _query_text(ds), ds["name"]
