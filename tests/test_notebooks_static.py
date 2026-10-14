"""Static checks on the notebooks themselves.

These cannot run Spark, but they can prove that the guarantees the project
claims are actually written into the notebooks: the leakage guard, the
Bronze→Silver reconciliation assertion, the absence of a random split, the
absence of a fake local serving fallback, and that no credential is hard-coded.
"""

from __future__ import annotations

import ast
import os
import re

import pytest

NOTEBOOK_HEADER = "# Databricks notebook source"

EXPECTED_NOTEBOOKS = [
    "00_config.py",
    "01_ingest_bronze.py",
    "02_transform_silver.py",
    "03_build_gold_features.py",
    "04_train_register_model.py",
    "05_deploy_and_test_endpoint.py",
    "06_generate_monitoring.py",
    "08_fraud_impact_simulation.py",
    "10_fraud_operations_timeseries.py",
]


def read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


@pytest.fixture(scope="module")
def sources(project_root):
    src = os.path.join(project_root, "src")
    return {name: read(os.path.join(src, name)) for name in EXPECTED_NOTEBOOKS}


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #

def test_all_expected_notebooks_exist(project_root):
    src = os.path.join(project_root, "src")
    for name in EXPECTED_NOTEBOOKS:
        assert os.path.isfile(os.path.join(src, name)), f"{name} is missing"


def test_notebooks_use_databricks_source_format(sources):
    for name, text in sources.items():
        assert text.startswith(NOTEBOOK_HEADER), (
            f"{name} must begin with '{NOTEBOOK_HEADER}' to import as a notebook"
        )


def test_notebooks_are_valid_python(notebook_paths):
    for path in notebook_paths:
        text = read(path)
        # `%run` / `%md` are Databricks magics, valid only inside a notebook.
        stripped = "\n".join(
            "" if line.lstrip().startswith(("%run", "%md", "%sql", "%pip"))
            else line
            for line in text.splitlines()
        )
        try:
            ast.parse(stripped)
        except SyntaxError as exc:
            pytest.fail(f"{os.path.basename(path)} has a syntax error: {exc}")


def test_notebooks_have_cell_separators(sources):
    for name, text in sources.items():
        assert "# COMMAND ----------" in text, f"{name} has no cell separators"


def test_pipeline_notebooks_import_the_shared_config(sources):
    for name, text in sources.items():
        if name == "00_config.py":
            continue
        assert "%run ./00_config" in text, f"{name} does not %run ./00_config"


def test_every_notebook_is_documented_with_markdown(sources):
    for name, text in sources.items():
        assert "# MAGIC %md" in text, f"{name} has no markdown documentation"


# --------------------------------------------------------------------------- #
# No secrets, no hard-coded workspace identity
# --------------------------------------------------------------------------- #

FORBIDDEN_PATTERNS = [
    (r"dapi[a-f0-9]{8,}", "a Databricks personal access token"),
    (r"https://[a-z0-9-]*\.cloud\.databricks\.com", "a hard-coded workspace URL"),
    (r"https://adb-\d+", "a hard-coded workspace URL"),
    (r"DATABRICKS_TOKEN\s*=\s*[\"'][^\"']+[\"']", "an inline token assignment"),
    (r"kaggle\.json", "a Kaggle credential file"),
    (r"[\"']sk-[A-Za-z0-9]{16,}[\"']", "an API key literal"),
]


def test_no_notebook_contains_a_credential_or_workspace_url(sources):
    for name, text in sources.items():
        for pattern, description in FORBIDDEN_PATTERNS:
            assert not re.search(pattern, text), f"{name} appears to contain {description}"


def test_no_source_file_contains_a_credential(project_root):
    # `tests/` is excluded on purpose: the sanitiser tests must contain
    # realistic token-shaped strings in order to prove they are scrubbed.
    for folder in ("src", "sql"):
        base = os.path.join(project_root, folder)
        for root, _dirs, filenames in os.walk(base):
            if "__pycache__" in root:
                continue
            for filename in filenames:
                if not filename.endswith((".py", ".sql")):
                    continue
                text = read(os.path.join(root, filename))
                for pattern, description in FORBIDDEN_PATTERNS:
                    assert not re.search(pattern, text), \
                        f"{filename} appears to contain {description}"


def test_the_current_user_is_read_from_the_session(sources):
    assert "current_user()" in sources["00_config.py"]


# --------------------------------------------------------------------------- #
# Serverless compatibility
#
# Free Edition offers only serverless (Spark Connect) compute, which rejects a
# handful of classic Spark APIs at *runtime*, not at deploy time. Each of these
# actually failed a real job run before being fixed.
# --------------------------------------------------------------------------- #

SERVERLESS_UNSUPPORTED = [
    (r"\.cache\(\)", ".cache() -> [NOT_SUPPORTED_WITH_SERVERLESS] PERSIST TABLE"),
    (r"\.persist\(", ".persist() -> [NOT_SUPPORTED_WITH_SERVERLESS] PERSIST TABLE"),
    (r"input_file_name\(\)", "input_file_name() -> unsupported on Spark Connect; "
                             "use _metadata.file_path"),
    (r"sparkContext", "SparkContext is not exposed on serverless"),
    (r"\.rdd\b", "RDD APIs are not available on serverless"),
]


def code_only(text: str) -> str:
    """Strip `#` comments (and therefore `# MAGIC %md` prose) from a notebook.

    Needed because the comments explaining *why* these APIs are banned
    necessarily name them.
    """
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def test_notebooks_avoid_apis_serverless_rejects(sources):
    for name, text in sources.items():
        body = code_only(text)
        for pattern, explanation in SERVERLESS_UNSUPPORTED:
            assert not re.search(pattern, body), f"{name} uses {explanation}"


def test_library_avoids_apis_serverless_rejects(project_root):
    package = os.path.join(project_root, "src", "fraud_common")
    for filename in sorted(os.listdir(package)):
        if not filename.endswith(".py"):
            continue
        body = code_only(read(os.path.join(package, filename)))
        for pattern, explanation in SERVERLESS_UNSUPPORTED:
            assert not re.search(pattern, body), f"{filename} uses {explanation}"


def test_the_serverless_scanner_actually_catches_a_violation():
    """Guard the guard: prove code_only() has not neutered the check."""
    offending = "df = spark.table('t').cache()\n"
    assert re.search(SERVERLESS_UNSUPPORTED[0][0], code_only(offending))


def test_timestamp_literals_avoid_the_ambiguous_z_suffix():
    """An ISO string ending in `Z` parses inconsistently across Spark versions."""
    from fraud_common import schema as schema_module

    assert not schema_module.SIMULATION_START_TS.endswith("Z")


# --------------------------------------------------------------------------- #
# Bronze
# --------------------------------------------------------------------------- #

def test_bronze_uses_an_explicit_schema_and_never_infers(sources):
    text = sources["01_ingest_bronze.py"]
    assert "SOURCE_SCHEMA_DDL" in text
    assert ".schema(READ_SCHEMA)" in text
    # The word appears in the prose explaining why it is avoided; what matters
    # is that it is never passed as a reader option.
    assert 'option("inferSchema"' not in text


def test_bronze_captures_malformed_records(sources):
    text = sources["01_ingest_bronze.py"]
    assert "columnNameOfCorruptRecord" in text
    assert "PERMISSIVE" in text
    assert "bronze_malformed_table" in text


def test_bronze_records_ingestion_lineage(sources):
    text = sources["01_ingest_bronze.py"]
    for column in ("ingestion_timestamp", "source_file", "batch_id"):
        assert column in text, f"bronze does not record {column}"


def test_bronze_appends_rather_than_overwrites(sources):
    assert 'mode("append")' in sources["01_ingest_bronze.py"]


def test_bronze_ingestion_is_idempotent(sources):
    text = sources["01_ingest_bronze.py"]
    assert "already_ingested" in text
    assert "compute_batch_id" in text


def test_bronze_asserts_the_source_column_set(sources):
    assert "source schema mismatch" in sources["01_ingest_bronze.py"]


# --------------------------------------------------------------------------- #
# Silver
# --------------------------------------------------------------------------- #

def test_silver_asserts_bronze_to_silver_reconciliation(sources):
    text = sources["02_transform_silver.py"]
    assert "assert bronze_count == silver_count + rejected_count" in text, (
        "Silver must assert that no row is lost between layers"
    )


def test_silver_preserves_rejected_records(sources):
    text = sources["02_transform_silver.py"]
    assert "silver_rejects_table" in text
    assert "reject_reasons" in text


def test_silver_persists_the_quality_results(sources):
    assert "silver_dq_table" in sources["02_transform_silver.py"]


def test_silver_checks_the_allowed_target_values(sources):
    text = sources["02_transform_silver.py"]
    assert "ALLOWED_TARGET_VALUES" in text
    assert "distinct_targets" in text


def test_silver_flags_duplicates_deterministically(sources):
    text = sources["02_transform_silver.py"]
    assert "occurrence_index" in text
    assert "is_exact_duplicate" in text
    assert "row_number()" in text


def test_silver_derives_time_fields_from_step(sources):
    text = sources["02_transform_silver.py"]
    for column in ("event_hour_of_day", "event_day_index", "event_timestamp"):
        assert column in text


# --------------------------------------------------------------------------- #
# Gold
# --------------------------------------------------------------------------- #

def test_gold_runs_the_leakage_guard(sources):
    text = sources["03_build_gold_features.py"]
    assert "assert_no_leakage" in text, "Gold must call the leakage guard at runtime"


def test_gold_documents_the_leakage_decisions(sources):
    text = sources["03_build_gold_features.py"]
    for marker in ("newbalanceOrig", "isFlaggedFraud", "post-transaction"):
        assert marker in text, f"Gold does not document the exclusion of {marker}"


def test_gold_verifies_the_chronological_split(sources):
    text = sources["03_build_gold_features.py"]
    assert "chronological violation" in text


def test_gold_persists_the_feature_contract(sources):
    assert "feature_contract_table" in sources["03_build_gold_features.py"]


def test_gold_excludes_only_valid_non_duplicate_rows(sources):
    assert "is_valid AND NOT is_exact_duplicate" in sources["03_build_gold_features.py"]


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def test_training_never_uses_a_random_split(sources):
    text = sources["04_train_register_model.py"]
    assert "train_test_split" not in text, "a random split is forbidden in this project"
    assert "compute_boundaries" in text


def test_training_asserts_chronological_ordering(sources):
    assert "assert_chronological" in sources["04_train_register_model.py"]


def test_training_selects_on_pr_auc(sources):
    text = sources["04_train_register_model.py"]
    assert "PRIMARY_METRIC" in text
    assert "compare_models" in text


def test_training_does_not_select_on_accuracy(sources):
    text = sources["04_train_register_model.py"].lower()
    assert "accuracy_score" not in text


def test_training_keeps_all_fraud_rows_when_sampling(sources):
    assert 'keep_all_when="is_fraud = 1"' in sources["04_train_register_model.py"]


def test_training_logs_signature_and_input_example(sources):
    text = sources["04_train_register_model.py"]
    assert "infer_signature" in text
    assert "input_example" in text


def test_training_logs_dataset_lineage(sources):
    assert "mlflow.data.from_pandas" in sources["04_train_register_model.py"]


def test_training_registers_into_unity_catalog(sources):
    text = sources["04_train_register_model.py"]
    assert 'mlflow.set_registry_uri("databricks-uc")' in text
    assert "register_model" in text
    assert "set_registered_model_alias" in text


def test_training_serves_probabilities_not_hard_labels(sources):
    assert 'pyfunc_predict_fn="predict_proba"' in sources["04_train_register_model.py"]


def test_training_chooses_the_threshold_on_validation(sources):
    text = sources["04_train_register_model.py"]
    assert "choose_threshold_for_max_fpr" in text
    assert "y_valid" in text.split("choose_threshold_for_max_fpr")[1][:200]


def test_training_fails_loudly_when_nothing_is_registered(sources):
    """No registered version anywhere -> hard failure, never a silent pass."""
    text = sources["04_train_register_model.py"]
    assert "raise RuntimeError(" in text
    assert "no existing version was" in text


def test_training_does_not_discard_valid_work_on_registration_failure(sources):
    """A failed registration must not throw away ETL, training and scoring.

    Free Edition denies s3:PutObject on the UC model location often enough that
    hard-failing here would block monitoring for no benefit -- but the run must
    still refuse to claim it produced a new registered model.
    """
    text = sources["04_train_register_model.py"]
    assert "reused_existing_version" in text
    assert "Do NOT describe this run as" in text


# --------------------------------------------------------------------------- #
# Serving
# --------------------------------------------------------------------------- #

def test_deployment_has_no_local_server_fallback(sources):
    """A local Flask/FastAPI stand-in must never masquerade as a deployment."""
    text = sources["05_deploy_and_test_endpoint.py"].lower()
    for forbidden in ("flask", "fastapi", "uvicorn", "http.server"):
        assert forbidden not in text, (
            f"05 references {forbidden}; Databricks serving must not be simulated"
        )


def test_deployment_creates_a_scale_to_zero_endpoint(sources):
    text = sources["05_deploy_and_test_endpoint.py"]
    assert "scale_to_zero_enabled" in text
    assert "build_endpoint_config" in text


def test_deployment_validates_the_response_structure(sources):
    text = sources["05_deploy_and_test_endpoint.py"]
    assert "parse_predictions" in text
    assert "validate_scores" in text


def test_verdict_is_only_verified_after_a_successful_invocation(sources):
    text = sources["05_deploy_and_test_endpoint.py"]
    verified_assignments = re.findall(r'VERDICT\s*=\s*"VERIFIED"', text)
    assert len(verified_assignments) == 1, "VERIFIED must be set in exactly one place"
    before, after = text.split('VERDICT = "VERIFIED"', 1)
    assert "invoke(" in before, "VERIFIED is set before any invocation happens"
    assert 'assert VERDICT == "VERIFIED"' in after


def test_deployment_sanitises_its_evidence(sources):
    text = sources["05_deploy_and_test_endpoint.py"]
    assert "sanitize_evidence" in text or "write_json_to_volume" in text
    assert "sanitize_error" in text


def test_deployment_prints_a_manual_resolution_path_on_failure(sources):
    text = sources["05_deploy_and_test_endpoint.py"]
    assert "Shortest resolution path" in text


def test_deployment_sends_a_real_data_request_too(sources):
    assert "real_gold_test_rows" in sources["05_deploy_and_test_endpoint.py"]


# --------------------------------------------------------------------------- #
# Monitoring
# --------------------------------------------------------------------------- #

def test_monitoring_separates_the_three_drift_families(sources):
    text = sources["06_generate_monitoring.py"]
    assert "drift.FEATURE_DRIFT" in text
    assert "drift.PREDICTION_DRIFT" in text
    assert "drift.PERFORMANCE" in text


def test_monitoring_states_that_feature_drift_is_not_degradation(sources):
    text = sources["06_generate_monitoring.py"].lower()
    assert "do not by themselves demonstrate model degradation" in text


def test_monitoring_models_delayed_labels(sources):
    text = sources["06_generate_monitoring.py"]
    assert "label_delay_windows" in text
    assert "label_available" in text


def test_monitoring_writes_no_performance_row_without_labels(sources):
    text = sources["06_generate_monitoring.py"]
    assert 'if not w["label_available"]:' in text
    assert "continue" in text


def test_monitoring_freezes_the_reference_binning(sources):
    text = sources["06_generate_monitoring.py"]
    assert "build_numeric_binning" in text
    assert "reference" in text.lower()


def test_monitoring_computes_every_required_statistic(sources):
    text = sources["06_generate_monitoring.py"]
    for required in ["psi", "missing_rate_delta", "predicted_positive_rate",
                     "avg_score", "transaction_volume", "pr_auc",
                     "precision", "recall"]:
        assert required in text, f"monitoring does not compute {required}"


def test_monitoring_records_which_scoring_path_was_used(sources):
    assert "scoring_path" in sources["06_generate_monitoring.py"]


# --------------------------------------------------------------------------- #
# Dashboard SQL
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def dashboard_sql(project_root):
    return read(os.path.join(project_root, "sql", "07_drift_dashboard.sql"))


def test_dashboard_covers_every_required_tile(dashboard_sql):
    for marker in [
        "Q1 · Transaction volume over time",
        "Q2 · Fraud-score distribution over time",
        "Q3 · Feature drift status by feature and window",
        "Q4 · Predicted-positive rate",
        "Q5 · Precision and recall where labels are available",
        "Q6 · Red / amber / green drift indicator",
    ]:
        assert marker in dashboard_sql, f"dashboard is missing: {marker}"


def test_dashboard_shows_unlabelled_windows_as_unknown(dashboard_sql):
    assert "AWAITING LABELS" in dashboard_sql
    assert "'UNKNOWN'" in dashboard_sql


def test_dashboard_warns_against_inferring_degradation_from_drift(dashboard_sql):
    assert "Degradation is NOT established" in dashboard_sql


def test_dashboard_reads_only_the_aggregate_tables(dashboard_sql):
    """Free Edition warehouses must not be pointed at the 6.3 M-row Gold table."""
    assert "gold_transaction_features" not in dashboard_sql
    assert "FROM workspace.fraud_poc.inference_log" not in dashboard_sql


def test_dashboard_includes_build_instructions(dashboard_sql):
    assert "HOW TO BUILD THE DASHBOARD" in dashboard_sql
    assert "SUGGESTED DASHBOARD LAYOUT" in dashboard_sql


# --------------------------------------------------------------------------- #
# Business impact simulation
#
# The failure mode this section guards is not a crash -- it is a plausible
# dollar figure presented without the assumptions that produced it, or without
# the caveat that it is a simulation. That is a reputational failure, and it is
# invisible in the output.
# --------------------------------------------------------------------------- #

def test_impact_notebook_states_the_simulation_disclaimer(sources):
    text = sources["08_fraud_impact_simulation.py"].lower()
    assert "simulated estimates" in text
    assert "production validation" in text
    assert "guaranteed real-world savings" in text


def test_impact_notebook_writes_the_assumptions_alongside_the_numbers(sources):
    text = sources["08_fraud_impact_simulation.py"]
    assert "ImpactAssumptions" in text
    assert "as_output_columns" in text or "assumptions" in text
    assert "impact_prevention_rate" in text


def test_impact_notebook_reads_its_assumptions_from_config_not_literals(sources):
    """Hard-coded costs in the notebook would silently ignore the job
    parameters and the widgets, so a stakeholder's re-run would not change."""
    text = sources["08_fraud_impact_simulation.py"]
    for field in ("CFG.impact_prevention_rate", "CFG.impact_cost_per_review",
                  "CFG.impact_customer_friction_cost",
                  "CFG.impact_analyst_daily_capacity"):
        assert field in text, f"08 does not read {field} from config"


def test_impact_notebook_simulates_on_held_out_rows_by_default(sources):
    """Simulating on the training period would quote savings on rows the model
    was fitted to."""
    text = sources["08_fraud_impact_simulation.py"]
    assert "impact_evaluation_split" in text
    assert "splits.TEST" in text


def test_impact_notebook_tolerates_alternative_probability_column_names(sources):
    text = sources["08_fraud_impact_simulation.py"]
    assert "COLUMN MAPPING" in text
    assert "SCORE_COLUMN_OVERRIDE" in text
    assert "resolve_column" in text


def test_impact_notebook_documents_its_input_contract(sources):
    """If the scored table is absent the notebook must say precisely what it
    needs, so the feature can be wired up later without reading the source."""
    text = sources["08_fraud_impact_simulation.py"]
    assert "EXPECTED_INPUT_CONTRACT" in text
    assert "required_columns" in text


def test_impact_notebook_writes_both_gold_tables(sources):
    text = sources["08_fraud_impact_simulation.py"]
    assert "fraud_impact_simulation_table" in text
    assert "fraud_impact_summary_table" in text


def test_impact_notebook_sweeps_thresholds_in_one_pass(sources):
    """One aggregation, not one query per threshold: nineteen scans of the
    scored table is nineteen times the Free Edition budget for identical
    arithmetic."""
    text = sources["08_fraud_impact_simulation.py"]
    assert "_threshold_aggregations" in text
    assert text.count(".agg(") == 1


# --------------------------------------------------------------------------- #
# Impact dashboard SQL
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def impact_sql(project_root):
    return read(os.path.join(project_root, "sql", "09_fraud_impact_dashboard.sql"))


def test_impact_dashboard_covers_every_required_section(impact_sql):
    for marker in [
        "SECTION A · EXECUTIVE KPI TILES",
        "SECTION B · THRESHOLD TRADE-OFF CHART",
        "SECTION C · FRAUD CAPTURED VS MISSED",
        "SECTION D · REVIEW WORKLOAD SIMULATION",
        "SECTION E · MODEL QUALITY VS BUSINESS VALUE",
        "SECTION F · FINAL BUSINESS RECOMMENDATION",
    ]:
        assert marker in impact_sql, f"impact dashboard is missing: {marker}"


def test_impact_dashboard_leads_with_the_disclaimer(impact_sql):
    head = impact_sql[:2500]
    assert "SIMULATED ESTIMATES" in head
    assert "production validation" in head


def test_impact_dashboard_shows_the_assumptions_on_the_page(impact_sql):
    assert "Assumptions in force" in impact_sql
    assert "assumption_prevention_rate" in impact_sql


def test_impact_dashboard_backticks_the_precision_column(impact_sql):
    """`precision` is an ANSI keyword; unbacktick'd it is a parse error."""
    assert "`precision`" in impact_sql
    assert not re.search(r"[^`\w]precision\s*\*", impact_sql.replace("`precision`", ""))


def test_impact_dashboard_reads_only_the_small_gold_tables(impact_sql):
    assert "gold_transaction_features" not in impact_sql
    assert "scored_predictions" not in impact_sql


def test_impact_dashboard_includes_build_instructions(impact_sql):
    assert "HOW TO BUILD THE DASHBOARD" in impact_sql
    assert "SUGGESTED DASHBOARD LAYOUT" in impact_sql


# --------------------------------------------------------------------------- #
# Operational replay
#
# Two honesty risks specific to a time axis: dates that look real but are not,
# and a replay that reads as a production record.
# --------------------------------------------------------------------------- #

def test_ops_notebook_says_the_dates_are_synthetic(sources):
    text = sources["10_fraud_operations_timeseries.py"]
    assert "SIMULATION_START_TS" in text
    assert "not real dates" in text.lower()


def test_ops_notebook_says_it_is_a_replay_not_a_production_record(sources):
    text = sources["10_fraud_operations_timeseries.py"].lower()
    assert "replay" in text
    assert "not a production record" in text
    assert "guaranteed real-world savings" in text


def test_ops_notebook_shares_the_time_derivation_with_silver(sources):
    """A second, independently written step->timestamp expression would put the
    two dashboards on subtly different clocks."""
    assert "step_to_timestamp_sql" in sources["10_fraud_operations_timeseries.py"]


def test_ops_notebook_freezes_the_threshold(sources):
    """Re-optimising the threshold per period would be hindsight no production
    desk has."""
    text = sources["10_fraud_operations_timeseries.py"]
    assert "_resolve_threshold" in text
    assert "THRESHOLD_SOURCE" in text
    assert "ops_threshold_override" in text


def test_ops_notebook_falls_back_when_no_recommendation_exists(sources):
    """It must run standalone, not hard-require notebook 08 to have run."""
    text = sources["10_fraud_operations_timeseries.py"]
    assert "CFG.operational_threshold" in text
    assert "fraud_impact_summary_table" in text


def test_ops_notebook_models_review_capacity(sources):
    text = sources["10_fraud_operations_timeseries.py"]
    assert "ops_analyst_headcount" in text
    assert "derive_operations_row" in text


def test_ops_notebook_warns_about_the_sampled_train_period(sources):
    """`scored_predictions` holds a SAMPLE of the train period, so plotting it
    on a volume chart would show a dip that is an artefact of sampling."""
    text = sources["10_fraud_operations_timeseries.py"]
    assert "artificially low volume" in text


def test_ops_notebook_writes_its_three_tables(sources):
    text = sources["10_fraud_operations_timeseries.py"]
    for table in ("fraud_operations_timeseries_table",
                  "fraud_operations_profile_table",
                  "fraud_operations_summary_table"):
        assert table in text, f"10 does not write {table}"


# --------------------------------------------------------------------------- #
# Operational dashboard SQL
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def ops_sql(project_root):
    return read(os.path.join(project_root, "sql", "11_fraud_operations_dashboard.sql"))


def test_ops_dashboard_covers_every_required_section(ops_sql):
    for marker in [
        "SECTION T · SINCE GO-LIVE HEADLINES",
        "SECTION U · MONEY OVER TIME — SAVINGS VS OPERATIONAL COST",
        "SECTION V · WHAT WE CAUGHT, OVER TIME",
        "SECTION W · CAN THE DESK KEEP UP?",
        "SECTION X · DETECTION QUALITY ON THE CLOCK",
    ]:
        assert marker in ops_sql, f"operations dashboard is missing: {marker}"


def test_ops_dashboard_leads_with_both_caveats(ops_sql):
    head = ops_sql[:2600]
    assert "THE DATES ARE SYNTHETIC" in head
    assert "REPLAY, NOT A PRODUCTION RECORD" in head
    assert "SIMULATED ESTIMATES" in head
    assert "production validation" in head


def test_ops_dashboard_uses_one_definition_of_operational_cost(ops_sql):
    assert "review cost + false-positive friction cost" in ops_sql
    assert "never a second one invented here" in ops_sql


def test_ops_dashboard_compares_savings_against_cost_over_time(ops_sql):
    assert "gross_estimated_savings" in ops_sql
    assert "operational_cost" in ops_sql
    assert "cumulative_net_savings" in ops_sql


def test_ops_dashboard_reads_only_the_operations_aggregates(ops_sql):
    assert "scored_predictions" not in ops_sql
    assert "gold_transaction_features" not in ops_sql


def test_ops_dashboard_includes_build_instructions(ops_sql):
    assert "HOW TO BUILD THE DASHBOARD" in ops_sql
    assert "SUGGESTED DASHBOARD LAYOUT" in ops_sql
