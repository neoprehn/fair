import pytest

from pyfair.utility.beta_pert import FairBetaPert
from pyfair.utility.fair_exception import FairException


def test_symmetric_mode_does_not_raise_and_matches_target_stdev():
    """mode == (low+high)/2 (mean == mode) must not raise and must produce
    the target stdev = range / (gamma + 2)."""
    low, mode, high, gamma = 0, 50, 100, 4
    pert = FairBetaPert(low, mode, high, gamma)
    target_stdev = (high - low) / (gamma + 2)
    assert pert._stdev == pytest.approx(target_stdev)


def test_boundary_modes_do_not_raise():
    """mode at the low or high boundary must not raise."""
    FairBetaPert(0, 0, 100)
    FairBetaPert(0, 100, 100)


def test_mode_outside_range_raises():
    """mode outside [low, high] must raise a FairException."""
    with pytest.raises(FairException):
        FairBetaPert(0, -1, 100)
    with pytest.raises(FairException):
        FairBetaPert(0, 101, 100)
