import pytest
from training import *
from transformer import *
from model import *


def test_save_text():
    """
    Test that previous text is saved properly.
    """
    model = Model("italian_numbers", "test.csv")
    model.generate_text("one")
    model.generate_text("two")
    model.generate_text("three")
    assert len(model._text) == 6
