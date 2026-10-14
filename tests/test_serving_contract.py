"""Serving payload shape, response validation, evidence sanitising, signature."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_common import features, schema, serving


# --------------------------------------------------------------------------- #
# Example payload
# --------------------------------------------------------------------------- #

def test_example_record_matches_the_contract_exactly():
    assert set(schema.EXAMPLE_SERVING_RECORD) == set(schema.served_features())


def test_example_payload_uses_the_documented_envelope():
    payload = serving.example_payload()
    assert set(payload) == {"dataframe_records"}
    assert serving.PAYLOAD_FORMAT == "dataframe_records"


def test_example_payload_preserves_signature_column_order():
    record = serving.example_payload()["dataframe_records"][0]
    assert list(record.keys()) == features.signature_input_columns()


def test_example_payload_can_request_several_rows():
    assert len(serving.example_payload(3)["dataframe_records"]) == 3


def test_example_record_uses_an_allowed_transaction_type():
    assert schema.EXAMPLE_SERVING_RECORD["type"] in schema.ALLOWED_TRANSACTION_TYPES


def test_example_record_carries_no_post_transaction_field():
    for forbidden in ("new_balance_orig", "new_balance_dest",
                      "is_fraud", "is_flagged_fraud", "step"):
        assert forbidden not in schema.EXAMPLE_SERVING_RECORD


# --------------------------------------------------------------------------- #
# Record validation
# --------------------------------------------------------------------------- #

def test_a_missing_feature_is_rejected():
    record = dict(schema.EXAMPLE_SERVING_RECORD)
    record.pop("amount")
    with pytest.raises(ValueError, match="missing required features"):
        serving.validate_record(record)


def test_an_extra_feature_is_rejected():
    record = dict(schema.EXAMPLE_SERVING_RECORD, new_balance_orig=0.0)
    with pytest.raises(ValueError, match="outside the contract"):
        serving.validate_record(record)


def test_a_null_feature_is_rejected():
    record = dict(schema.EXAMPLE_SERVING_RECORD, amount=None)
    with pytest.raises(ValueError, match="null"):
        serving.validate_record(record)


def test_a_wrong_type_is_rejected():
    record = dict(schema.EXAMPLE_SERVING_RECORD, type=42)
    with pytest.raises(TypeError):
        serving.validate_record(record)


def test_an_unknown_transaction_type_is_rejected():
    record = dict(schema.EXAMPLE_SERVING_RECORD, type="CRYPTO_SWAP")
    with pytest.raises(ValueError, match="outside"):
        serving.validate_record(record)


def test_an_empty_batch_is_rejected():
    with pytest.raises(ValueError, match="at least one record"):
        serving.build_payload([])


# --------------------------------------------------------------------------- #
# Payload from a frame
# --------------------------------------------------------------------------- #

def test_payload_from_frame_produces_json_native_types(synthetic_gold):
    payload = serving.payload_from_frame(synthetic_gold, limit=3)
    records = payload["dataframe_records"]
    assert len(records) == 3
    for record in records:
        for name, dtype in schema.served_dtypes().items():
            value = record[name]
            # numpy scalars break json.dumps inside the SDK
            assert not isinstance(value, np.generic), f"{name} is a numpy scalar"
            if dtype == "int":
                assert isinstance(value, int)
            elif dtype == "double":
                assert isinstance(value, float)
            else:
                assert isinstance(value, str)


def test_payload_from_frame_reorders_to_the_signature(synthetic_gold):
    shuffled = synthetic_gold[list(reversed(schema.served_features()))]
    record = serving.payload_from_frame(shuffled, limit=1)["dataframe_records"][0]
    assert list(record.keys()) == features.signature_input_columns()


def test_payload_from_frame_rejects_a_frame_missing_features(synthetic_gold):
    with pytest.raises(KeyError, match="missing served features"):
        serving.payload_from_frame(synthetic_gold.drop(columns=["amount"]))


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #

def test_a_well_formed_response_is_parsed():
    predictions = serving.parse_predictions({"predictions": [[0.9, 0.1]]}, 1)
    assert predictions == [[0.9, 0.1]]


def test_a_response_without_predictions_is_rejected():
    with pytest.raises(ValueError, match="no 'predictions' key"):
        serving.parse_predictions({"error_code": "BAD_REQUEST"}, 1)


def test_a_row_count_mismatch_is_rejected():
    with pytest.raises(ValueError, match="expected 3 predictions"):
        serving.parse_predictions({"predictions": [0.1]}, 3)


def test_a_non_list_predictions_field_is_rejected():
    with pytest.raises(TypeError):
        serving.parse_predictions({"predictions": 0.5}, 1)


@pytest.mark.parametrize("predictions,expected", [
    ([[0.9, 0.1], [0.2, 0.8]], [0.1, 0.8]),      # predict_proba pairs
    ([0.1, 0.8], [0.1, 0.8]),                     # scalar per row
    ([{"0": 0.9, "1": 0.1}], [0.1]),              # dict-shaped output
    ([[0.42]], [0.42]),                           # single-column vector
])
def test_scores_are_extracted_from_every_supported_shape(predictions, expected):
    assert serving.extract_scores(predictions) == pytest.approx(expected)


def test_an_uninterpretable_prediction_shape_is_rejected():
    with pytest.raises(ValueError):
        serving.extract_scores([[0.1, 0.2, 0.3]])


def test_scores_outside_zero_to_one_are_rejected():
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        serving.validate_scores([1.4])


def test_nan_scores_are_rejected():
    with pytest.raises(ValueError, match="NaN"):
        serving.validate_scores([float("nan")])


def test_valid_scores_pass():
    serving.validate_scores([0.0, 0.5, 1.0])      # must not raise


# --------------------------------------------------------------------------- #
# Endpoint configuration
# --------------------------------------------------------------------------- #

def test_endpoint_config_is_scale_to_zero_cpu():
    config = serving.build_endpoint_config(
        "paysim-fraud-poc", "workspace.fraud_poc.paysim_fraud_clf", "1"
    )
    entity = config["config"]["served_entities"][0]
    assert entity["scale_to_zero_enabled"] is True
    assert entity["workload_type"] == "CPU"
    assert entity["entity_name"] == "workspace.fraud_poc.paysim_fraud_clf"
    assert entity["entity_version"] == "1"


def test_traffic_routes_to_the_served_entity():
    config = serving.build_endpoint_config(
        "e", "workspace.fraud_poc.paysim_fraud_clf", "2"
    )
    route = config["config"]["traffic_config"]["routes"][0]
    assert route["served_model_name"] == config["config"]["served_entities"][0]["name"]
    assert route["traffic_percentage"] == 100


# --------------------------------------------------------------------------- #
# Endpoint readiness — regression test for a real production failure
# --------------------------------------------------------------------------- #

def test_not_ready_is_not_ready():
    """Regression: `"READY" in "NOT_READY"` is True.

    That substring test reported a still-provisioning endpoint as ready, and
    the first query then failed with a misleading ResourceDoesNotExist.
    """
    assert serving.is_endpoint_ready("NOT_READY") is False
    assert serving.is_endpoint_ready("READY") is True


@pytest.mark.parametrize("value", [
    "READY", "EndpointStateReady.READY", "ready", " Ready ",
])
def test_ready_is_recognised_in_every_spelling(value):
    assert serving.is_endpoint_ready(value) is True


@pytest.mark.parametrize("value", [
    "NOT_READY", "EndpointStateReady.NOT_READY", None, "", "PROVISIONING",
])
def test_non_ready_states_are_rejected(value):
    assert serving.is_endpoint_ready(value) is False


@pytest.mark.parametrize("value,expected", [
    ("UPDATE_FAILED", True),
    ("EndpointStateConfigUpdate.UPDATE_FAILED", True),
    ("UPDATE_CANCELED", True),
    ("IN_PROGRESS", False),
    ("UPDATE_SUCCEEDED", False),
    (None, False),
])
def test_update_failure_states(value, expected):
    assert serving.is_endpoint_update_failed(value) is expected


def test_in_progress_is_neither_ready_nor_failed():
    """The state the real endpoint was in when the bug fired."""
    assert serving.is_endpoint_ready("NOT_READY") is False
    assert serving.is_endpoint_update_failed("IN_PROGRESS") is False


def test_enum_like_objects_are_handled():
    class FakeEnum:
        value = "READY"

        def __str__(self):
            return "EndpointStateReady.READY"

    assert serving.is_endpoint_ready(FakeEnum()) is True


def test_a_two_level_model_name_is_rejected():
    with pytest.raises(ValueError, match="three-level"):
        serving.build_endpoint_config("e", "schema.model", "1")


def test_an_empty_endpoint_name_is_rejected():
    with pytest.raises(ValueError):
        serving.build_endpoint_config("", "a.b.c", "1")


# --------------------------------------------------------------------------- #
# Evidence sanitising — nothing credential-shaped may be persisted
# --------------------------------------------------------------------------- #

def test_secret_looking_keys_are_redacted():
    evidence = serving.sanitize_evidence({
        "Authorization": "Bearer abc123",
        "api_key": "sk-live-xyz",
        "token": "dapi0123456789abcdef",
        "endpoint_name": "paysim-fraud-poc",
    })
    assert evidence["Authorization"] == serving.REDACTED
    assert evidence["api_key"] == serving.REDACTED
    assert evidence["token"] == serving.REDACTED
    assert evidence["endpoint_name"] == "paysim-fraud-poc"


def test_bearer_tokens_inside_free_text_are_scrubbed():
    cleaned = serving.scrub_text("failed: Bearer eyJhbGciOiJIUzI1NiJ9 rejected")
    assert "eyJhbGciOiJIUzI1NiJ9" not in cleaned
    assert serving.REDACTED in cleaned


def test_databricks_personal_access_tokens_are_scrubbed():
    # Synthetic fixture assembled at runtime; no credential is stored here.
    dummy_token = "dapi" + "1234567890abcdef" * 2
    cleaned = serving.scrub_text(f"using {dummy_token}")
    assert dummy_token not in cleaned
    assert serving.REDACTED in cleaned


def test_credentials_embedded_in_a_url_are_scrubbed():
    cleaned = serving.scrub_text("https://user:hunter2@example.cloud.databricks.com/api")
    assert "hunter2" not in cleaned


def test_sanitising_reaches_nested_structures():
    evidence = serving.sanitize_evidence(
        {"stages": [{"detail": {"client_secret": "s3cr3t", "rows": 5}}]}
    )
    assert evidence["stages"][0]["detail"]["client_secret"] == serving.REDACTED
    assert evidence["stages"][0]["detail"]["rows"] == 5


def test_sanitising_preserves_ordinary_values():
    payload = serving.example_payload()
    assert serving.sanitize_evidence(payload) == payload


def test_sanitised_errors_carry_type_and_scrubbed_message():
    error = serving.sanitize_error(RuntimeError("denied for Bearer abc.def-123"))
    assert error["error_type"] == "RuntimeError"
    assert "abc.def-123" not in error["error_message"]


def test_error_messages_are_truncated():
    error = serving.sanitize_error(RuntimeError("x" * 5000), max_chars=100)
    assert len(error["error_message"]) == 100


# --------------------------------------------------------------------------- #
# MLflow signature round-trip
# --------------------------------------------------------------------------- #

def test_mlflow_signature_matches_the_serving_payload(synthetic_gold):
    """The logged signature must accept exactly the payload we send.

    Skipped when MLflow is absent locally; the CI/Databricks run has it.
    """
    mlflow_models = pytest.importorskip("mlflow.models")

    frame = synthetic_gold[schema.served_features()].head(50)
    output = np.column_stack([np.full(50, 0.9), np.full(50, 0.1)])
    signature = mlflow_models.infer_signature(frame, output)

    signature_columns = [spec.name for spec in signature.inputs.inputs]
    assert signature_columns == features.signature_input_columns()

    payload_record = serving.payload_from_frame(frame, limit=1)["dataframe_records"][0]
    assert list(payload_record.keys()) == signature_columns


def test_signature_input_types_are_what_the_contract_declares(synthetic_gold):
    mlflow_models = pytest.importorskip("mlflow.models")

    frame = synthetic_gold[schema.served_features()].head(10)
    signature = mlflow_models.infer_signature(
        frame, np.column_stack([np.full(10, 0.5), np.full(10, 0.5)])
    )
    declared = schema.served_dtypes()
    for spec in signature.inputs.inputs:
        # MLflow renders DataType as "DataType.double"; take the leaf name.
        inferred = str(spec.type).rsplit(".", 1)[-1]
        if declared[spec.name] == "string":
            assert inferred == "string", f"{spec.name}: {inferred}"
        else:
            assert inferred in {"double", "float", "long", "integer"}, \
                f"{spec.name}: {inferred}"


def test_the_signature_carries_no_leaking_column(synthetic_gold):
    mlflow_models = pytest.importorskip("mlflow.models")

    frame = synthetic_gold[schema.served_features()].head(10)
    signature = mlflow_models.infer_signature(
        frame, np.column_stack([np.full(10, 0.5), np.full(10, 0.5)])
    )
    schema.assert_no_leakage([spec.name for spec in signature.inputs.inputs])


# --------------------------------------------------------------------------- #
# Training / serving parity
# --------------------------------------------------------------------------- #

def test_pandas_derivation_produces_exactly_the_served_features(synthetic_silver):
    derived = features.derive_features_pandas(synthetic_silver)
    assert list(derived.columns) == features.signature_input_columns()


def test_derived_features_are_all_serialisable(synthetic_silver):
    derived = features.derive_features_pandas(synthetic_silver)
    payload = serving.payload_from_frame(derived, limit=5)
    assert len(payload["dataframe_records"]) == 5


def test_derivation_rejects_a_frame_without_the_pre_transaction_inputs():
    with pytest.raises(KeyError, match="missing Silver input columns"):
        features.derive_features_pandas(pd.DataFrame({"amount": [1.0]}))
