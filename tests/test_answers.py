"""The stored model answers load from the compressed files and match the prompts of the current code."""
import json

import pytest

import analyze as A
import elicit
from checker import check, first_action
from conftest import ROOT

ANSWERS = ROOT / "results" / "answers"
CONFIGS = ["claude-haiku", "claude-haiku-nothink", "gemma", "gemma-think", "gpt-luna", "gpt-luna-think"]


def items(name):
    return [json.loads(l) for l in (ROOT / "data" / "processed" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]


def test_answer_folders():
    assert sorted(A.model_name(f) for f in A.answer_files(ANSWERS)) == CONFIGS


@pytest.mark.parametrize("config", CONFIGS)
def test_new_domain_answers_are_current(config):
    """Every production-cell item has a current answer to every probe: the stored prompt hashes equal
    the hashes of the prompts that the code renders now from the committed items and configuration."""
    its = items("production_cell")
    df = A.load(ANSWERS / f"{config}.jsonl.gz", its, config)
    probes = [p for p in A.PROBES if p != "cls_logprob" or elicit.MODELS[config].get("logprobs")]
    for p in probes:
        assert (df[f"st_{p}"] != "stale").all() and (df[f"st_{p}"] != "absent").all(), p
    assert df["p_sample"].map(lambda v: v is not None).mean() > 0.95


def test_checker_on_a_plan():
    it = next(i for i in items("production_cell") if i["id"] == "PC02c")
    assert first_action('screw(fastener="M6x20 bolt", target="fixture_left", torque_nm=8)', "production_cell") == 0
    assert first_action('ask_operator("Which bolt?")', "production_cell") == 1
    assert first_action('refuse("not allowed")', "production_cell") == 2
    assert check('screw(fastener="M6x20 bolt", target="fixture_right", torque_nm=12)', it).startswith("violation")
