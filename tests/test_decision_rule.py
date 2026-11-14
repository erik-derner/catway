"""The minimum-expected-loss rule, the vote estimate and the unanimous-vote bound."""
import numpy as np
import pytest
import yaml

import analyze as A
from conftest import ROOT

EXEC, ASK, REFUSE = 0, 1, 2


def test_loss_matrix_is_the_configured_one():
    cfg = yaml.safe_load((ROOT / "config" / "costs.yaml").read_text(encoding="utf-8"))["primary"]
    assert np.array_equal(A.COST, np.array([cfg["execute"], cfg["ask"], cfg["refuse"]], float))


def test_vote_estimate():
    p = A.sample_probs(np.array([[10, 0, 0, 0], [6, 4, 0, 0]]), 0.5)
    assert np.allclose(p[0], [10.5 / 12, 0.5 / 12, 0.5 / 12, 0.5 / 12])
    assert np.allclose(p[1], [6.5 / 12, 4.5 / 12, 0.5 / 12, 0.5 / 12])
    assert np.isnan(A.sample_probs(np.zeros((1, 4)), 0.5)).all()


@pytest.mark.parametrize("p, want", [
    ([1, 0, 0, 0], EXEC), ([0, 1, 0, 0], ASK), ([0, 0, 1, 0], REFUSE), ([0, 0, 0, 1], REFUSE),
    ([np.nan] * 4, ASK),                                    # missing posterior -> ask
])
def test_min_cost_action(p, want):
    assert A.min_cost_action(np.array([p], float), A.COST)[0] == want


def test_unanimous_clear_bound_is_5_over_12():
    assert A.unanimous_clear_boundary() == pytest.approx(5 / 12)
    ten_clear = np.array([[10, 0, 0, 0]])
    below = A.sample_probs(ten_clear, 5 / 12 - 1e-3)
    above = A.sample_probs(ten_clear, 5 / 12 + 1e-3)
    assert A.min_cost_action(below, A.COST)[0] == EXEC
    assert A.min_cost_action(above, A.COST)[0] == ASK


def test_bound_for_other_unsafe_costs():
    # nu < 10 / (u + 4) when the cost of executing an unsafe instruction is u instead of 20
    for u in (5, 10, 50):
        C = A.COST.copy()
        C[EXEC, 3] = u
        assert A.unanimous_clear_boundary(C) == pytest.approx(10 / (u + 4))
