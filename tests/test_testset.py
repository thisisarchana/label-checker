from app.rules.label import LabelData
from app.rules.models import Verdict
from app.rules.run_checks import overall_verdict, run_all_checks
from eval.evaluate import run_cases, summarize
from eval.generate_testset import CLEAN_PRODUCTS, build_cases


def test_every_clean_product_passes():
    for product in CLEAN_PRODUCTS:
        findings = run_all_checks(LabelData(**product["label"]))
        assert overall_verdict(findings) == Verdict.PASS, product["id"]


def test_generation_is_deterministic():
    assert build_cases() == build_cases()


def test_all_violation_types_are_present():
    types = {c["violation_type"] for c in build_cases()}
    assert types == {
        "none", "allergen_not_declared", "protein_claim_unsupported",
        "false_high_protein_added", "gluten_grain_in_gf_product",
        "gluten_ppm_too_high", "hidden_allergen_ingredient",
        "natural_claim_no_rule",
    }


def test_case_ids_are_unique():
    ids = [c["id"] for c in build_cases()]
    assert len(ids) == len(set(ids))


def test_rule_engine_has_no_false_passes_on_testset():
    summary = summarize(run_cases(build_cases()))
    assert summary["false_passes"] == []
    assert summary["false_alarms"] == []