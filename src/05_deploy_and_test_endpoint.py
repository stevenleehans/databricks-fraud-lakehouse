# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Deploy to Model Serving and invoke it
# MAGIC
# MAGIC Creates a **scale-to-zero CPU** serving endpoint for the registered Unity
# MAGIC Catalog model, waits for it to become ready, and sends a real request.
# MAGIC
# MAGIC ## Honesty rules this notebook enforces
# MAGIC
# MAGIC * `VERDICT` is set to `VERIFIED` only after an HTTP response was received
# MAGIC   **and** its prediction structure validated. Anything else is `FAILED` or
# MAGIC   `BLOCKED`, with the exact non-secret error captured.
# MAGIC * There is no local fallback server. If Databricks serving is unavailable
# MAGIC   this notebook fails and tells you why — it never simulates a deployment.
# MAGIC * Authentication is handled entirely by the Databricks SDK's ambient
# MAGIC   credentials. No token is read, printed, stored or passed by this code,
# MAGIC   and all evidence is written through `serving.sanitize_evidence`.
# MAGIC
# MAGIC ## Free Edition note
# MAGIC Model Serving availability on Free Edition varies, and CPU endpoints are
# MAGIC quota-limited. If creation is refused, the failure path below prints the
# MAGIC shortest manual resolution.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import json
import time

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

VERDICT = "NOT_ATTEMPTED"
EVIDENCE = {
    "generated_at": utc_now_iso(),
    "endpoint_name": CFG.endpoint_name,
    "model_name": CFG.uc_model_name,
    "verdict": VERDICT,
    "stages": {},
}


def record(stage: str, status: str, **details) -> None:
    EVIDENCE["stages"][stage] = {"status": status, **serving.sanitize_evidence(details)}
    log("serve", f"{stage}: {status}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Resolve the model version to serve
# MAGIC
# MAGIC Prefer the `Champion` alias; fall back to the highest version number if the
# MAGIC alias could not be set (a restricted workspace may allow registration but
# MAGIC not aliasing).

# COMMAND ----------

MODEL_VERSION = None
VERSION_SOURCE = None

try:
    version_info = client.get_model_version_by_alias(
        CFG.uc_model_name, CFG.champion_alias
    )
    MODEL_VERSION = str(version_info.version)
    VERSION_SOURCE = f"alias:{CFG.champion_alias}"
except Exception as exc:
    log("serve", f"alias lookup failed ({type(exc).__name__}); trying latest version")
    try:
        versions = client.search_model_versions(f"name = '{CFG.uc_model_name}'")
        if not versions:
            raise LookupError(
                f"no versions found for {CFG.uc_model_name}. Run notebook 04 first."
            )
        latest = max(versions, key=lambda v: int(v.version))
        MODEL_VERSION = str(latest.version)
        VERSION_SOURCE = "latest_version"
    except Exception as inner:
        record("resolve_model_version", "FAILED", **serving.sanitize_error(inner))
        VERDICT = "BLOCKED"
        raise RuntimeError(
            f"Cannot resolve a version of {CFG.uc_model_name} to deploy. "
            "Notebook 04 must complete and register a model first."
        ) from inner

record("resolve_model_version", "OK", version=MODEL_VERSION, source=VERSION_SOURCE)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Build the endpoint configuration programmatically

# COMMAND ----------

ENDPOINT_CONFIG = serving.build_endpoint_config(
    endpoint_name=CFG.endpoint_name,
    uc_model_name=CFG.uc_model_name,
    model_version=MODEL_VERSION,
    workload_size=CFG.endpoint_workload_size,
    scale_to_zero=CFG.endpoint_scale_to_zero,
)
record("build_config", "OK", config=ENDPOINT_CONFIG)
print(json.dumps(ENDPOINT_CONFIG, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Create or update the endpoint

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)

w = WorkspaceClient()          # ambient auth; this notebook never touches a token

served_entity = ServedEntityInput(
    name=ENDPOINT_CONFIG["config"]["served_entities"][0]["name"],
    entity_name=CFG.uc_model_name,
    entity_version=MODEL_VERSION,
    workload_size=CFG.endpoint_workload_size,
    scale_to_zero_enabled=CFG.endpoint_scale_to_zero,
)

endpoint_exists = False
try:
    w.serving_endpoints.get(CFG.endpoint_name)
    endpoint_exists = True
except Exception:
    endpoint_exists = False

try:
    if endpoint_exists:
        log("serve", f"endpoint {CFG.endpoint_name} exists — updating to "
                     f"version {MODEL_VERSION}")
        w.serving_endpoints.update_config(
            name=CFG.endpoint_name, served_entities=[served_entity]
        )
        record("create_or_update", "UPDATED", version=MODEL_VERSION)
    else:
        log("serve", f"creating endpoint {CFG.endpoint_name} …")
        w.serving_endpoints.create(
            name=CFG.endpoint_name,
            config=EndpointCoreConfigInput(served_entities=[served_entity]),
        )
        record("create_or_update", "CREATED", version=MODEL_VERSION)
except Exception as exc:
    error = serving.sanitize_error(exc)
    record("create_or_update", "FAILED", **error)
    VERDICT = "BLOCKED"
    EVIDENCE["verdict"] = VERDICT
    log("serve", "ENDPOINT CREATION FAILED — this is NOT a deployment.")
    log("serve", f"  {error['error_type']}: {error['error_message']}")
    log("serve", "Shortest resolution path:")
    log("serve", "  1. Workspace > Serving. If the tab is absent, Model Serving "
                 "is not enabled for this workspace/edition — the pipeline stops "
                 "here and everything before it still stands.")
    log("serve", "  2. If you see a quota error, delete any other endpoint "
                 "(Serving > … > Delete) and re-run this notebook.")
    log("serve", f"  3. If you see a permission error: GRANT EXECUTE ON MODEL "
                 f"{CFG.uc_model_name} TO `<your-principal>`;")
    log("serve", "  4. If the model version is PENDING_REGISTRATION, wait for the "
                 "UC scan to finish and re-run.")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Wait for readiness
# MAGIC
# MAGIC A cold scale-to-zero endpoint needs several minutes on first creation.

# COMMAND ----------

def wait_for_ready(timeout_minutes: int) -> dict:
    deadline = time.time() + timeout_minutes * 60
    last_state = None
    while time.time() < deadline:
        detail = w.serving_endpoints.get(CFG.endpoint_name)
        state = detail.state
        ready = str(getattr(state, "ready", "")) if state else ""
        update = str(getattr(state, "config_update", "")) if state else ""
        current = f"ready={ready} config_update={update}"
        if current != last_state:
            log("serve", f"endpoint state: {current}")
            last_state = current
        # Exact state matching, via the tested helpers. A substring test would
        # treat NOT_READY as ready ("READY" in "NOT_READY") and query an
        # endpoint that is still provisioning.
        if serving.is_endpoint_ready(getattr(state, "ready", None)):
            return {"ready": True, "state": current}
        if serving.is_endpoint_update_failed(getattr(state, "config_update", None)):
            return {"ready": False, "state": current, "reason": "config update failed"}
        time.sleep(20)
    return {"ready": False, "state": last_state or "unknown", "reason": "timeout"}


readiness = wait_for_ready(CFG.endpoint_ready_timeout_minutes)
if not readiness["ready"]:
    record("wait_for_ready", "FAILED", **readiness)
    VERDICT = "FAILED"
    EVIDENCE["verdict"] = VERDICT
    raise RuntimeError(
        f"Endpoint {CFG.endpoint_name} did not become READY within "
        f"{CFG.endpoint_ready_timeout_minutes} minutes (state: {readiness['state']}). "
        "Check Workspace > Serving > "
        f"{CFG.endpoint_name} > Events for the build log. Do NOT report this as "
        "a successful deployment."
    )
record("wait_for_ready", "OK", **readiness)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Build a request that matches the logged signature
# MAGIC
# MAGIC Two payloads are sent: the canonical fabricated example (also in the
# MAGIC README) and a batch drawn from real Gold **test-period** rows — real data
# MAGIC catches encoding mismatches a hand-written example never will.

# COMMAND ----------

spark_session = get_spark()

example_payload = serving.example_payload(n=1)
print("example payload:")
print(json.dumps(example_payload, indent=2))

# Deliberately a *mixed* sample. A plain LIMIT returns whichever rows the first
# partition holds -- in practice five consecutive fraud rows, which all land in
# the same leaf and return byte-identical scores. That cannot distinguish a
# working endpoint from one returning a constant.
test_gold = spark_session.table(CFG.gold_features_table).where("split_label = 'test'")
sample_pdf = (
    test_gold.where("is_fraud = 0").limit(3)
    .unionByName(test_gold.where("is_fraud = 1").limit(2))
    .select(*schema.served_features(), "is_fraud")
    .toPandas()
)
EXPECTED_LABELS = sample_pdf["is_fraud"].astype(int).tolist()
real_rows_pdf = sample_pdf[schema.served_features()]
real_payload = serving.payload_from_frame(real_rows_pdf, limit=len(real_rows_pdf))
log("serve", f"real sample labels (ground truth): {EXPECTED_LABELS}")
record("build_payload", "OK",
       example_rows=1, real_rows=len(real_payload[serving.PAYLOAD_FORMAT]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Invoke and validate

# COMMAND ----------

def invoke(payload: dict, label: str) -> dict:
    """Send one request through the SDK and validate the response structure."""
    records = payload[serving.PAYLOAD_FORMAT]
    started = time.time()
    response = w.serving_endpoints.query(
        name=CFG.endpoint_name, dataframe_records=records
    )
    latency_ms = (time.time() - started) * 1000.0

    body = response.as_dict() if hasattr(response, "as_dict") else dict(response)
    predictions = serving.parse_predictions(body, expected_rows=len(records))
    scores = serving.extract_scores(predictions)
    serving.validate_scores(scores)

    result = {
        "label": label,
        "request_rows": len(records),
        "response_rows": len(predictions),
        "latency_ms": round(latency_ms, 1),
        "raw_prediction_sample": predictions[:2],
        "fraud_scores": [round(s, 6) for s in scores],
        "predicted_positive_at_threshold": [
            bool(s >= CFG.operational_threshold) for s in scores
        ],
    }
    log("serve", f"{label}: {len(predictions)} predictions in "
                 f"{latency_ms:.0f} ms, scores={result['fraud_scores']}")
    return result


try:
    example_result = invoke(example_payload, "canonical_example")
    real_result = invoke(real_payload, "real_gold_test_rows")

    # Attach ground truth so the evidence shows the endpoint *discriminates*
    # rather than merely responding.
    real_result["ground_truth_labels"] = EXPECTED_LABELS
    scores = real_result["fraud_scores"]
    legit_scores = [s for s, y in zip(scores, EXPECTED_LABELS) if y == 0]
    fraud_scores = [s for s, y in zip(scores, EXPECTED_LABELS) if y == 1]
    real_result["mean_score_legit"] = (
        round(sum(legit_scores) / len(legit_scores), 6) if legit_scores else None
    )
    real_result["mean_score_fraud"] = (
        round(sum(fraud_scores) / len(fraud_scores), 6) if fraud_scores else None
    )
    real_result["separates_classes"] = bool(
        legit_scores and fraud_scores and max(legit_scores) < min(fraud_scores)
    )
    real_result["distinct_scores"] = len(set(scores))
    log("serve", f"legit mean={real_result['mean_score_legit']} "
                 f"fraud mean={real_result['mean_score_fraud']} "
                 f"separates={real_result['separates_classes']}")

    if real_result["distinct_scores"] == 1:
        log("serve", "WARNING: every row returned the same score. The endpoint "
                     "responded, but this sample cannot show discrimination.")

    record("invoke", "OK", results=[example_result, real_result])
    VERDICT = "VERIFIED"
except Exception as exc:
    error = serving.sanitize_error(exc)
    record("invoke", "FAILED", **error)
    VERDICT = "FAILED"
    EVIDENCE["verdict"] = VERDICT
    log("serve", "INVOCATION FAILED — the endpoint exists but did not return a "
                 "valid prediction. This is NOT a verified deployment.")
    log("serve", f"  {error['error_type']}: {error['error_message']}")
    log("serve", "Most likely cause: the request columns do not match the logged "
                 "signature. Compare the payload above with "
                 f"{CFG.feature_contract_table} (served_to_endpoint = true).")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · Save sanitised evidence
# MAGIC
# MAGIC Request bodies, scores, latency and endpoint state only. No headers, no
# MAGIC URLs with credentials, no tokens — `sanitize_evidence` redacts anything
# MAGIC credential-shaped before this touches disk.

# COMMAND ----------

EVIDENCE["verdict"] = VERDICT
EVIDENCE["model_version"] = MODEL_VERSION
EVIDENCE["version_source"] = VERSION_SOURCE
EVIDENCE["scale_to_zero"] = CFG.endpoint_scale_to_zero
EVIDENCE["workload_size"] = CFG.endpoint_workload_size
EVIDENCE["example_request_body"] = example_payload

try:
    write_json_to_volume("serving_evidence.json", EVIDENCE)
except Exception as exc:
    log("serve", f"could not persist evidence: {type(exc).__name__}")

print(json.dumps(serving.sanitize_evidence(EVIDENCE), indent=2, default=str))

assert VERDICT == "VERIFIED", f"deployment not verified (verdict={VERDICT})"
log("serve", f"DONE — endpoint '{CFG.endpoint_name}' serving "
             f"{CFG.uc_model_name} v{MODEL_VERSION}: VERIFIED")
