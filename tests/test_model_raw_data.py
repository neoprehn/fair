import numpy as np
import pandas as pd
import pytest

from pyfair import FairModel
from pyfair.utility.fair_exception import FairException


def test_input_raw_data_accepts_ndarray():
    """Regression test: a genuine np.ndarray must be accepted.

    The type check previously compared against np.array (the array-creation
    function) instead of np.ndarray (the type), so type(array) was never in
    the accepted-types list and every real ndarray was rejected.
    """
    model = FairModel(name="raw_ndarray", n_simulations=5)
    array = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    model.input_raw_data('Loss Magnitude', array)

    assert list(model.export_results()['Loss Magnitude']) == list(array)


def test_input_raw_data_accepts_list():
    model = FairModel(name="raw_list", n_simulations=3)

    model.input_raw_data('Loss Magnitude', [1.0, 2.0, 3.0])

    assert list(model.export_results()['Loss Magnitude']) == [1.0, 2.0, 3.0]


def test_input_raw_data_accepts_series():
    model = FairModel(name="raw_series", n_simulations=3)

    model.input_raw_data('Loss Magnitude', pd.Series([1.0, 2.0, 3.0]))

    assert list(model.export_results()['Loss Magnitude']) == [1.0, 2.0, 3.0]


def test_input_raw_data_rejects_wrong_type():
    model = FairModel(name="raw_wrong_type", n_simulations=3)

    with pytest.raises(FairException):
        model.input_raw_data('Loss Magnitude', "not an array")


def test_input_raw_data_rejects_wrong_length():
    model = FairModel(name="raw_wrong_length", n_simulations=3)

    with pytest.raises(FairException):
        model.input_raw_data('Loss Magnitude', np.array([1.0, 2.0]))
