"""Serving payload construction, response validation and evidence sanitising.

The endpoint request schema is derived from the same feature contract used to
build Gold and to log the MLflow signature, so a mismatch is a test failure
rather than a 400 discovered during the demo.

Nothing in this module ever reads, stores or prints a token. Evidence written
to disk passes through :func:`sanitize_evidence`, which drops headers, URLs
with embedded credentials and any key that looks secret-ish.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .features import signature_input_columns
from .schema import ALLOWED_TRANSACTION_TYPES, EXAMPLE_SERVING_RECORD, served_dtypes

# Databricks Model Serving accepts several payload shapes; `dataframe_records`
# is the most readable for a demo and maps 1:1 onto the pandas signature.
PAYLOAD_FORMAT = "dataframe_records"

_SECRET_KEY_PATTERN = re.compile(
    r"(token|secret|password|passwd|authorization|auth|api[-_]?key|credential|"
    r"bearer|cookie|session|client[-_]?secret|private[-_]?key)",
    re.IGNORECASE,
)
_BEARER_PATTERN = re.compile(r"Bearer\s+[A-Za-z0-9._\-]+", re.IGNORECASE)
_DAPI_PATTERN = re.compile(r"dapi[a-f0-9]{8,}", re.IGNORECASE)
_URL_CREDENTIAL_PATTERN = re.compile(r"://[^/\s:@]+:[^/\s@]+@")

REDACTED = "<redacted>"


# --------------------------------------------------------------------------- #
# Request construction
# --------------------------------------------------------------------------- #

def validate_record(record: Mapping[str, Any]) -> None:
    """Check one request record against the online feature contract."""
    expected = signature_input_columns()
    missing = [c for c in expected if c not in record]
    extra = [c for c in record if c not in expected]
    if missing:
        raise ValueError(f"serving record is missing required features: {missing}")
    if extra:
        raise ValueError(
            f"serving record has features outside the contract: {extra}. "
            "The endpoint signature rejects unknown columns."
        )

    dtypes = served_dtypes()
    for name, dtype in dtypes.items():
        value = record[name]
        if value is None:
            raise ValueError(f"feature {name!r} is null; the signature requires a value")
        if dtype == "string":
            if not isinstance(value, str):
                raise TypeError(f"feature {name!r} must be a string, got {type(value).__name__}")
        elif dtype == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"feature {name!r} must be an int, got {type(value).__name__}")
        elif dtype == "double":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"feature {name!r} must be numeric, got {type(value).__name__}")

    if record["type"] not in ALLOWED_TRANSACTION_TYPES:
        raise ValueError(
            f"type={record['type']!r} is outside {ALLOWED_TRANSACTION_TYPES}"
        )


def build_payload(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Build a Databricks Model Serving request body.

    Column order is normalised to the signature order; every record is
    validated before it leaves the client.
    """
    if not records:
        raise ValueError("at least one record is required")
    order = signature_input_columns()
    normalised: List[Dict[str, Any]] = []
    for rec in records:
        validate_record(rec)
        normalised.append({k: rec[k] for k in order})
    return {PAYLOAD_FORMAT: normalised}


def payload_from_frame(df, limit: Optional[int] = None) -> Dict[str, Any]:
    """Build a payload from a pandas frame of *served features*."""
    order = signature_input_columns()
    missing = [c for c in order if c not in df.columns]
    if missing:
        raise KeyError(f"frame is missing served features: {missing}")
    frame = df[order]
    if limit is not None:
        frame = frame.head(limit)
    records = []
    for rec in frame.to_dict(orient="records"):
        clean: Dict[str, Any] = {}
        for name, dtype in served_dtypes().items():
            value = rec[name]
            if dtype == "int":
                clean[name] = int(value)
            elif dtype == "double":
                clean[name] = float(value)
            else:
                clean[name] = str(value)
        records.append(clean)
    return build_payload(records)


def example_payload(n: int = 1) -> Dict[str, Any]:
    """Canonical example request used in the README and the smoke test."""
    return build_payload([dict(EXAMPLE_SERVING_RECORD) for _ in range(n)])


# --------------------------------------------------------------------------- #
# Response validation
# --------------------------------------------------------------------------- #

def parse_predictions(response_json: Mapping[str, Any], expected_rows: int) -> List[Any]:
    """Validate the response envelope and return the prediction list.

    Databricks wraps model output as ``{"predictions": [...]}``. Anything else
    -- including a 200 that carries an error body -- is treated as a failure so
    a broken endpoint can never be reported as a success.
    """
    if not isinstance(response_json, Mapping):
        raise TypeError(f"expected a JSON object, got {type(response_json).__name__}")
    if "predictions" not in response_json:
        keys = sorted(response_json.keys())
        raise ValueError(
            f"response has no 'predictions' key (keys present: {keys}). "
            "This usually means the endpoint returned an error envelope."
        )
    preds = response_json["predictions"]
    if not isinstance(preds, list):
        raise TypeError(f"'predictions' must be a list, got {type(preds).__name__}")
    if len(preds) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} predictions, got {len(preds)}"
        )
    return preds


def extract_scores(predictions: Sequence[Any]) -> List[float]:
    """Normalise prediction shapes into fraud probabilities.

    Handles the three shapes a scikit-learn model can produce through MLflow:
    a scalar per row, ``{"0": p0, "1": p1}``-style dicts, and ``[p0, p1]``
    probability pairs.
    """
    scores: List[float] = []
    for p in predictions:
        if isinstance(p, Mapping):
            if "1" in p:
                scores.append(float(p["1"]))
            elif 1 in p:
                scores.append(float(p[1]))
            elif len(p) == 1:
                scores.append(float(next(iter(p.values()))))
            else:
                raise ValueError(f"cannot interpret prediction mapping: {sorted(p)}")
        elif isinstance(p, (list, tuple)):
            if len(p) == 2:
                scores.append(float(p[1]))
            elif len(p) == 1:
                scores.append(float(p[0]))
            else:
                raise ValueError(f"cannot interpret prediction vector of length {len(p)}")
        elif isinstance(p, bool):
            scores.append(float(p))
        elif isinstance(p, (int, float)):
            scores.append(float(p))
        else:
            raise TypeError(f"unsupported prediction element type: {type(p).__name__}")
    return scores


def validate_scores(scores: Iterable[float]) -> None:
    for s in scores:
        if s != s:
            raise ValueError("prediction score is NaN")
        if not 0.0 <= s <= 1.0:
            raise ValueError(
                f"prediction score {s} is outside [0, 1]; the served model must "
                "expose a calibrated probability, not a raw decision value"
            )


# --------------------------------------------------------------------------- #
# Endpoint configuration
# --------------------------------------------------------------------------- #

READY_STATES = {"READY"}
FAILED_UPDATE_STATES = {"UPDATE_FAILED", "UPDATE_CANCELED"}


def _state_leaf(value: object) -> str:
    """Normalise an SDK enum or string to its bare state name.

    The SDK may hand back `EndpointStateReady.NOT_READY`, a plain
    `"NOT_READY"`, or an enum whose `str()` includes the class. Take the last
    dotted component and upper-case it.
    """
    if value is None:
        return ""
    text = getattr(value, "value", None) or str(value)
    return text.rsplit(".", 1)[-1].strip().upper()


def is_endpoint_ready(value: object) -> bool:
    """True only when the endpoint is actually serving.

    Deliberately an exact match. Substring tests are a trap here: `"READY" in
    "NOT_READY"` is True, which reports a still-provisioning endpoint as ready
    and produces a misleading `ResourceDoesNotExist` on the first query.
    """
    return _state_leaf(value) in READY_STATES


def is_endpoint_update_failed(value: object) -> bool:
    return _state_leaf(value) in FAILED_UPDATE_STATES


def build_endpoint_config(
    endpoint_name: str,
    uc_model_name: str,
    model_version: str,
    workload_size: str = "Small",
    scale_to_zero: bool = True,
    entity_traffic_percentage: int = 100,
) -> Dict[str, Any]:
    """Serving endpoint spec for the Databricks SDK / REST API.

    ``scale_to_zero`` is mandatory in spirit for Free Edition: an always-on
    endpoint burns the quota and will be reaped.
    """
    if not endpoint_name:
        raise ValueError("endpoint_name is required")
    if uc_model_name.count(".") != 2:
        raise ValueError(
            f"expected a three-level Unity Catalog name, got {uc_model_name!r}"
        )
    entity_name_short = uc_model_name.split(".")[-1]
    return {
        "name": endpoint_name,
        "config": {
            "served_entities": [
                {
                    "name": f"{entity_name_short}-{model_version}",
                    "entity_name": uc_model_name,
                    "entity_version": str(model_version),
                    "workload_size": workload_size,
                    "workload_type": "CPU",
                    "scale_to_zero_enabled": bool(scale_to_zero),
                }
            ],
            "traffic_config": {
                "routes": [
                    {
                        "served_model_name": f"{entity_name_short}-{model_version}",
                        "traffic_percentage": int(entity_traffic_percentage),
                    }
                ]
            },
        },
    }


# --------------------------------------------------------------------------- #
# Evidence sanitising
# --------------------------------------------------------------------------- #

def scrub_text(text: str) -> str:
    out = _BEARER_PATTERN.sub(f"Bearer {REDACTED}", text)
    out = _DAPI_PATTERN.sub(REDACTED, out)
    out = _URL_CREDENTIAL_PATTERN.sub(f"://{REDACTED}@", out)
    return out


def sanitize_evidence(obj: Any) -> Any:
    """Recursively strip anything credential-shaped before writing to disk.

    Applied to every artifact ``05_deploy_and_test_endpoint`` saves. Keys whose
    name matches a secret pattern are replaced wholesale; string values are
    scrubbed for embedded bearer tokens and `dapi…` personal access tokens.
    """
    if isinstance(obj, Mapping):
        clean: Dict[str, Any] = {}
        for key, value in obj.items():
            if isinstance(key, str) and _SECRET_KEY_PATTERN.search(key):
                clean[key] = REDACTED
            else:
                clean[key] = sanitize_evidence(value)
        return clean
    if isinstance(obj, (list, tuple)):
        return [sanitize_evidence(v) for v in obj]
    if isinstance(obj, str):
        return scrub_text(obj)
    return obj


def sanitize_error(exc: BaseException, max_chars: int = 1200) -> Dict[str, str]:
    """Turn an exception into a non-secret, reportable record."""
    message = scrub_text(str(exc))[:max_chars]
    return {"error_type": type(exc).__name__, "error_message": message}
