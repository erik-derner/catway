"""The stored supplementary tables are complete and agree with the stored pipeline tables."""
import re

import pandas as pd

import supplementary_analyses as S
from conftest import ROOT

OUT = ROOT / "results" / "supplementary_analyses"


def test_every_listed_table_is_stored():
    """Every CSV named in the module's list of outputs is in results/supplementary_analyses/."""
    listed = set(re.findall(r"\b(\w+\.csv)\b", S.__doc__.split("Outputs")[1]))
    assert listed and listed <= {f.name for f in OUT.glob("*.csv")}


def test_validation_passes():
    """Every recomputed value equals the stored analyze.py / method_study.py tables (validation.csv)."""
    chk = pd.read_csv(OUT / "validation.csv")
    assert len(chk) > 0 and chk.ok.all(), chk[~chk.ok].check.tolist()
