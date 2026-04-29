"""Tests for the calculator module."""

from calculator import add, subtract, multiply, divide, average
import pytest


def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0


def test_subtract():
    assert subtract(5, 3) == 2


def test_multiply():
    assert multiply(3, 4) == 12


def test_divide():
    assert divide(10, 2) == 5.0
    assert divide(7, 2) == 3.5


def test_divide_by_zero():
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        divide(1, 0)


def test_average():
    assert average([1, 2, 3]) == 2.0
    assert average([10]) == 10.0


def test_average_empty():
    with pytest.raises(ValueError, match="Cannot average an empty list"):
        average([])
