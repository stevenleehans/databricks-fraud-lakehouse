"""Source schema, target values and the no-leakage guarantee."""

from __future__ import annotations

import pytest

from fraud_common import features, schema


# --------------------------------------------------------------------------- #
# Source schema
# --------------------------------------------------------------------------- #

def test_source_has_the_eleven_published_columns():
    assert len(schema.SOURCE_COLUMNS) == 11


def test_source_column_spelling_matches_kaggle_exactly():
    # PaySim really does ship `oldbalanceOrg` (no 'i') next to `newbalanceOrig`
    # (with 'i'). Normalising this away would silently break the CSV read.
    assert "oldbalanceOrg" in schema.SOURCE_COLUMNS
    assert "newbalanceOrig" in schema.SOURCE_COLUMNS
    assert "oldbalanceOrig" not in schema.SOURCE_COLUMNS
    assert "newbalanceOrg" not in schema.SOURCE_COLUMNS


def test_ddl_declares_every_source_column_once():
    ddl_fields = [part.strip().split()[0] for part in schema.SOURCE_SCHEMA_DDL.split(",")]
    assert ddl_fields == schema.SOURCE_COLUMNS


def test_ddl_types_are_explicit_and_sensible():
    types = {
        part.strip().split()[0]: part.strip().split()[1]
        for part in schema.SOURCE_SCHEMA_DDL.split(",")
    }
    assert types["step"] == "INT"
    assert types["amount"] == "DOUBLE"
    assert types["type"] == "STRING"
    assert types["isFraud"] == "INT"


def test_pandas_dtypes_cover_every_source_column():
    assert set(schema.SOURCE_PANDAS_DTYPES) == set(schema.SOURCE_COLUMNS)


def test_rename_map_covers_every_source_column():
    assert set(features.SOURCE_TO_SILVER) == set(schema.SOURCE_COLUMNS)
    assert len(set(features.SOURCE_TO_SILVER.values())) == 11


# --------------------------------------------------------------------------- #
# Target
# --------------------------------------------------------------------------- #

def test_target_is_binary():
    assert schema.TARGET_COLUMN == "isFraud"
    assert schema.ALLOWED_TARGET_VALUES == [0, 1]


def test_allowed_transaction_types_are_the_five_paysim_categories():
    assert sorted(schema.ALLOWED_TRANSACTION_TYPES) == [
        "CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"
    ]


def test_fraud_capable_types_are_a_subset_of_allowed_types():
    assert set(schema.FRAUD_CAPABLE_TYPES) <= set(schema.ALLOWED_TRANSACTION_TYPES)


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #

def test_no_post_transaction_column_is_served():
    for spec in schema.GOLD_COLUMNS:
        if spec.served:
            assert spec.availability == schema.PRE_TRANSACTION, (
                f"{spec.name} is served but marked {spec.availability}"
            )


@pytest.mark.parametrize(
    "forbidden",
    ["is_fraud", "isFraud", "is_flagged_fraud", "isFlaggedFraud",
     "new_balance_orig", "newbalanceOrig", "new_balance_dest", "newbalanceDest",
     "nameOrig", "nameDest", "name_orig_hash", "name_dest_hash"],
)
def test_assert_no_leakage_rejects_forbidden_column(forbidden):
    with pytest.raises(ValueError, match="leakage"):
        schema.assert_no_leakage(schema.served_features() + [forbidden])


def test_assert_no_leakage_accepts_the_real_feature_list():
    schema.assert_no_leakage(schema.served_features())      # must not raise


def test_target_is_not_in_the_served_features():
    assert "is_fraud" not in schema.served_features()
    assert "isFraud" not in schema.served_features()


def test_step_is_retained_but_not_served():
    spec = schema.get_spec("step")
    assert spec.served is False, "step must not be a model input"
    assert "step" in schema.gold_column_names(), "step is needed for splitting"


def test_raw_identifiers_never_reach_the_endpoint():
    served = set(schema.served_features())
    assert not served & {"name_orig", "name_dest", "name_orig_hash", "name_dest_hash"}


def test_dest_is_merchant_is_served_and_documented_as_account_type():
    spec = schema.get_spec("dest_is_merchant")
    assert spec.served and spec.availability == schema.PRE_TRANSACTION
    # It is derived from nameDest, so the rationale must be recorded.
    assert "merchant" in spec.description.lower()


# --------------------------------------------------------------------------- #
# Contract consistency
# --------------------------------------------------------------------------- #

def test_schema_and_features_agree_on_the_served_list():
    features.assert_feature_contract_consistent()


def test_served_features_are_unique_and_ordered_like_the_signature():
    served = schema.served_features()
    assert len(served) == len(set(served))
    assert served == features.signature_input_columns()


def test_numeric_and_categorical_partition_the_served_features():
    numeric = set(schema.numeric_features())
    categorical = set(schema.categorical_features())
    assert numeric.isdisjoint(categorical)
    assert numeric | categorical == set(schema.served_features())


def test_only_type_is_categorical():
    assert schema.categorical_features() == ["type"]


def test_every_gold_column_is_documented():
    for spec in schema.GOLD_COLUMNS:
        assert spec.description.strip(), f"{spec.name} has no description"
        assert spec.derivation, f"{spec.name} has no derivation recorded"
        assert spec.availability in {
            schema.PRE_TRANSACTION, schema.POST_TRANSACTION,
            schema.LABEL, schema.IDENTIFIER,
        }


def test_gold_column_names_are_unique():
    names = schema.gold_column_names()
    assert len(names) == len(set(names))


# --------------------------------------------------------------------------- #
# Quality rules
# --------------------------------------------------------------------------- #

def test_quality_rules_have_valid_severities():
    for rule in schema.QUALITY_RULES:
        assert rule.severity in {"reject", "warn"}
        assert rule.expression.strip()
        assert rule.description.strip()


def test_reject_rules_cover_target_type_and_amount():
    covered = " ".join(r.expression for r in schema.reject_rules())
    assert "is_fraud IN (0, 1)" in covered
    assert "amount >= 0" in covered
    assert "type IN" in covered


def test_silver_valid_expression_combines_every_reject_rule():
    expression = schema.silver_valid_expression()
    # Rules are parenthesised before being joined, so ") AND (" counts the
    # joins without also counting the ANDs inside individual rules.
    assert expression.count(") AND (") == len(schema.reject_rules()) - 1
    for rule in schema.reject_rules():
        assert rule.expression in expression


def test_rule_names_are_unique():
    names = [r.name for r in schema.QUALITY_RULES]
    assert len(names) == len(set(names))
