import pytest


@pytest.mark.parametrize('test_input, expected', [
    ((1, 2), 3),
    ((-1, 2), 1),
    ((1, -2), -1),
    ((0, 0), 0),
    ((-10, -10), -20)])
def test_add(test_input, expected):
    assert test_input[0] + test_input[1] == expected
