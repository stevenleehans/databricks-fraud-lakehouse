"""Feature derivations (hand-computed) and configuration loading."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from fraud_common import features, schema
from fraud_common.config import Config, load_config, resolve_experiment_name, validate_config


# --------------------------------------------------------------------------- #
# Derivations — checked against values worked out by hand
# --------------------------------------------------------------------------- #

@pytest.fixture()
def one_row():
    """step 26 -> day 2, hour 1. A drained, over-balance transfer to a merchant."""
    return pd.DataFrame({
        "step": [26],
        "type": ["TRANSFER"],
        "amount": [1000.0],
        "old_balance_orig": [1000.0],
        "old_balance_dest": [0.0],
        "name_dest": ["M1979787155"],
    })


def test_hour_of_day_is_zero_based(one_row):
    assert features.derive_features_pandas(one_row)["hour_of_day"].iloc[0] == 1


@pytest.mark.parametrize("step,hour", [(1, 0), (24, 23), (25, 0), (744, 23)])
def test_hour_of_day_wraps_correctly(step, hour):
    frame = pd.DataFrame({
        "step": [step], "type": ["PAYMENT"], "amount": [1.0],
        "old_balance_orig": [1.0], "old_balance_dest": [0.0], "name_dest": ["C1"],
    })
    assert features.derive_features_pandas(frame)["hour_of_day"].iloc[0] == hour


@pytest.mark.parametrize("step,expected", [(1, 1), (5, 1), (6, 1), (7, 0), (24, 0)])
def test_is_night_covers_midnight_to_six(step, expected):
    frame = pd.DataFrame({
        "step": [step], "type": ["PAYMENT"], "amount": [1.0],
        "old_balance_orig": [1.0], "old_balance_dest": [0.0], "name_dest": ["C1"],
    })
    assert features.derive_features_pandas(frame)["is_night"].iloc[0] == expected


def test_day_of_week_cycles_weekly():
    frame = pd.DataFrame({
        "step": [1, 25, 49, 169],       # days 1, 2, 3, 8
        "type": ["PAYMENT"] * 4, "amount": [1.0] * 4,
        "old_balance_orig": [1.0] * 4, "old_balance_dest": [0.0] * 4,
        "name_dest": ["C1"] * 4,
    })
    assert features.derive_features_pandas(frame)["day_of_week"].tolist() == [0, 1, 2, 0]


def test_drains_origin_account_detects_a_full_sweep(one_row):
    assert features.derive_features_pandas(one_row)["drains_origin_account"].iloc[0] == 1


def test_drains_origin_account_is_zero_for_an_empty_account():
    """A zero-balance origin cannot be 'drained' — that is a different signal."""
    frame = pd.DataFrame({
        "step": [1], "type": ["TRANSFER"], "amount": [500.0],
        "old_balance_orig": [0.0], "old_balance_dest": [0.0], "name_dest": ["C1"],
    })
    derived = features.derive_features_pandas(frame)
    assert derived["drains_origin_account"].iloc[0] == 0
    assert derived["orig_balance_is_zero"].iloc[0] == 1
    assert derived["exceeds_origin_balance"].iloc[0] == 1


def test_amount_to_balance_ratio_uses_the_epsilon(one_row):
    derived = features.derive_features_pandas(one_row)
    assert derived["amount_to_balance_ratio"].iloc[0] == pytest.approx(1000.0 / 1001.0)


def test_ratio_is_capped_for_zero_balance_accounts():
    frame = pd.DataFrame({
        "step": [1], "type": ["TRANSFER"], "amount": [1e9],
        "old_balance_orig": [0.0], "old_balance_dest": [0.0], "name_dest": ["C1"],
    })
    ratio = features.derive_features_pandas(frame)["amount_to_balance_ratio"].iloc[0]
    assert ratio == pytest.approx(features.MAX_AMOUNT_RATIO)
    assert np.isfinite(ratio)


def test_ratio_never_divides_by_zero(synthetic_silver):
    derived = features.derive_features_pandas(synthetic_silver)
    assert np.isfinite(derived["amount_to_balance_ratio"]).all()


def test_dest_is_merchant_reads_the_m_prefix(one_row):
    assert features.derive_features_pandas(one_row)["dest_is_merchant"].iloc[0] == 1


def test_customer_destinations_are_not_merchants():
    frame = pd.DataFrame({
        "step": [1], "type": ["TRANSFER"], "amount": [1.0],
        "old_balance_orig": [1.0], "old_balance_dest": [0.0], "name_dest": ["C1979787155"],
    })
    assert features.derive_features_pandas(frame)["dest_is_merchant"].iloc[0] == 0


def test_amount_log_is_log1p(one_row):
    derived = features.derive_features_pandas(one_row)
    assert derived["amount_log"].iloc[0] == pytest.approx(np.log1p(1000.0))


def test_log_of_a_zero_amount_is_zero_not_infinite():
    frame = pd.DataFrame({
        "step": [1], "type": ["PAYMENT"], "amount": [0.0],
        "old_balance_orig": [100.0], "old_balance_dest": [0.0], "name_dest": ["C1"],
    })
    assert features.derive_features_pandas(frame)["amount_log"].iloc[0] == 0.0


def test_derivations_are_finite_across_the_whole_frame(synthetic_silver):
    derived = features.derive_features_pandas(synthetic_silver)
    for column in schema.numeric_features():
        assert np.isfinite(derived[column].astype(float)).all(), f"{column} has non-finite values"


def test_derivation_is_deterministic(synthetic_silver):
    a = features.derive_features_pandas(synthetic_silver)
    b = features.derive_features_pandas(synthetic_silver)
    pd.testing.assert_frame_equal(a, b)


# --------------------------------------------------------------------------- #
# SQL / pandas parity of the declarations
# --------------------------------------------------------------------------- #

def test_every_served_feature_has_a_sql_expression():
    assert set(features.FEATURE_SQL) == set(schema.served_features())


def test_gold_projection_emits_every_gold_column():
    aliases = {expr.rsplit(" AS ", 1)[1] for expr in features.gold_select_expressions()}
    assert aliases == set(schema.gold_column_names())


def test_no_served_expression_reads_a_post_transaction_column():
    """A served feature may only reference pre-transaction Silver inputs."""
    forbidden = ["new_balance_orig", "new_balance_dest", "is_fraud", "is_flagged_fraud",
                 "name_orig"]
    for name, expression in features.FEATURE_SQL.items():
        for column in forbidden:
            assert column not in expression, \
                f"served feature {name!r} references {column!r}"


def test_only_the_merchant_test_touches_the_destination_name():
    users = [n for n, e in features.FEATURE_SQL.items() if "name_dest" in e]
    assert users == ["dest_is_merchant"]


def test_identifiers_are_hashed_in_the_passthrough_projection():
    assert features.PASSTHROUGH_SQL["name_orig_hash"].startswith("SHA2(")
    assert features.PASSTHROUGH_SQL["name_dest_hash"].startswith("SHA2(")


def test_the_contract_self_check_passes():
    features.assert_feature_contract_consistent()


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

def test_defaults_load_and_validate():
    cfg = load_config()
    validate_config(cfg)
    assert cfg.catalog and cfg.schema


def test_derived_names_are_three_level_unity_catalog_names():
    cfg = load_config()
    assert cfg.uc_model_name.count(".") == 2
    for table in [cfg.bronze_table, cfg.silver_table, cfg.gold_features_table,
                  cfg.inference_log_table, cfg.feature_drift_table,
                  cfg.prediction_drift_table, cfg.performance_table]:
        assert table.count(".") == 2, table
        assert table.startswith(f"{cfg.catalog}.{cfg.schema}.")


def test_table_names_are_unique():
    cfg = load_config()
    names = [
        cfg.bronze_table, cfg.bronze_audit_table, cfg.bronze_malformed_table,
        cfg.silver_table, cfg.silver_rejects_table, cfg.silver_dq_table,
        cfg.gold_features_table, cfg.feature_contract_table,
        cfg.inference_log_table, cfg.reference_stats_table,
        cfg.feature_drift_table, cfg.prediction_drift_table,
        cfg.score_histogram_table, cfg.performance_table, cfg.window_dim_table,
    ]
    assert len(names) == len(set(names))


def test_no_default_leaks_a_workspace_url_or_identity():
    for key, value in load_config().to_dict().items():
        if not isinstance(value, str):
            continue
        lowered = value.lower()
        assert "databricks.com" not in lowered, key
        assert "https://" not in lowered, key
        assert "@" not in value, key
        assert "dapi" not in lowered, key


def test_experiment_name_is_resolved_from_the_session_not_hard_coded():
    cfg = load_config()
    assert cfg.experiment_name is None
    assert resolve_experiment_name(cfg, "someone@example.com") == \
        "/Users/someone@example.com/paysim_fraud_poc"


def test_experiment_name_falls_back_when_the_user_is_unknown():
    assert resolve_experiment_name(load_config(), "").startswith("/Shared/")


def test_an_explicit_experiment_name_wins():
    cfg = load_config({"experiment_name": "/Shared/custom"})
    assert resolve_experiment_name(cfg, "a@b.com") == "/Shared/custom"


def test_endpoint_defaults_are_free_edition_friendly():
    cfg = load_config()
    assert cfg.endpoint_scale_to_zero is True
    assert cfg.endpoint_workload_size == "Small"


def test_overrides_are_coerced_from_widget_strings():
    cfg = load_config({
        "random_seed": "7",
        "train_negative_sample_fraction": "0.25",
        "endpoint_scale_to_zero": "false",
    })
    assert cfg.random_seed == 7
    assert cfg.train_negative_sample_fraction == 0.25
    assert cfg.endpoint_scale_to_zero is False


def test_an_empty_string_clears_an_optional_field():
    assert load_config({"experiment_name": ""}).experiment_name is None


def test_an_unknown_field_is_rejected():
    with pytest.raises(KeyError):
        load_config({"not_a_field": "x"})


def test_environment_variables_are_honoured(monkeypatch):
    monkeypatch.setenv("FRAUD_CATALOG", "main")
    monkeypatch.setenv("FRAUD_RANDOM_SEED", "99")
    cfg = load_config()
    assert cfg.catalog == "main"
    assert cfg.random_seed == 99


def test_explicit_overrides_beat_the_environment(monkeypatch):
    monkeypatch.setenv("FRAUD_CATALOG", "from_env")
    assert load_config({"catalog": "from_arg"}).catalog == "from_arg"


@pytest.mark.parametrize("bad", [
    {"train_step_fraction": 0.9, "valid_step_fraction": 0.2},   # no test split
    {"train_negative_sample_fraction": 0.0},
    {"train_negative_sample_fraction": 1.5},
    {"psi_amber_threshold": 0.5, "psi_red_threshold": 0.25},
    {"psi_bins": 1},
    {"operational_threshold": 1.5},
    {"monitoring_window_count": 0},
    {"target_max_fpr": 1.0},
])
def test_invalid_configurations_are_rejected(bad):
    with pytest.raises(ValueError):
        load_config(bad)


def test_describe_does_not_print_anything_secret():
    text = load_config().describe().lower()
    for marker in ("token", "password", "secret", "dapi"):
        assert marker not in text


def test_evidence_dir_lives_under_the_volume():
    cfg = load_config()
    assert cfg.evidence_dir.startswith(cfg.volume_root)


def test_config_is_immutable():
    with pytest.raises(Exception):
        load_config().catalog = "other"


def test_input_path_default_points_at_a_volume():
    assert load_config().input_path.startswith("/Volumes/")


def test_every_config_field_is_overridable_by_environment():
    from dataclasses import fields

    from fraud_common.config import ENV_PREFIX

    assert ENV_PREFIX == "FRAUD_"
    names = {f.name for f in fields(Config)}
    assert {"catalog", "schema", "input_path", "random_seed"} <= names
