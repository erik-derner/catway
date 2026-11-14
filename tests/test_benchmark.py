"""The authored domains rebuild to the committed item files; checksums and data files are consistent."""
import hashlib
import json

import pytest
import yaml

import build_data as B
from conftest import ROOT

PROCESSED = ROOT / "data" / "processed"


def committed(name):
    return [json.loads(l) for l in (PROCESSED / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.mark.parametrize("name, speaker", [("production_cell", "operator"), ("mobile_robot", "staff")])
def test_new_domains_rebuild_exactly(name, speaker):
    items = B.build_ood(name, speaker)
    assert len(items) == 100
    assert sorted(it["condition"] for it in items) == ["ambiguous"] * 25 + ["clear"] * 25 + ["infeasible"] * 25 + ["unsafe"] * 25
    assert items == committed(name)
    assert B.round_two(items) == committed(f"{name}_r2")


def test_round_two_resolves_ambiguous_items_only():
    r2 = committed("production_cell_r2")
    base = {it["id"]: it for it in committed("production_cell")}
    for it in r2:
        b = base[it["meta"]["base"]]
        assert it["condition"] == ("clear" if b["condition"] == "ambiguous" else b["condition"])
        assert it["text"] == b["text"]
        assert "The robot asked:" in it["situation"]


def test_checksums_of_committed_files():
    sums = dict(reversed(l.split("  ")) for l in (PROCESSED / "SHA256SUMS").read_text(encoding="utf-8").splitlines())
    assert len(sums) == 9
    for name, digest in sums.items():
        path = PROCESSED / name
        if path.exists():                       # household*.jsonl exist only after a rebuild
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name


def test_household_infeasible_items_hold_no_ambik_text():
    spec = yaml.safe_load((ROOT / "data" / "items" / "household_infeasible.yaml").read_text(encoding="utf-8"))
    assert sum(v.get("ok") is not False for v in spec) == 150
    for v in spec:
        assert "based_on" not in v
        assert ("text" in v) != ("text_from_ambik" in v), v["id"]
        ends = [e for s, e, _ in v.get("text_from_ambik", [])]
        starts = [s for s, e, _ in v.get("text_from_ambik", [])]
        assert all(s <= e for s, e in zip(starts, ends)) and starts == sorted(starts)


def test_restore_ambik_texts():
    import pandas as pd
    amb = pd.DataFrame({"id": [7], "unambiguous_direct": ["Put the red cup on the table. "]})
    spec = [{"id": "IM7", "ambik_id": 7, "text_from_ambik": [[8, 11, "blue"]]}, {"id": "IM8", "text": "x"}]
    B.restore_ambik_texts(spec, amb)
    assert spec[0]["text"] == "Put the blue cup on the table."
    assert spec[0]["based_on"] == "Put the red cup on the table."
    assert spec[1] == {"id": "IM8", "text": "x"}
