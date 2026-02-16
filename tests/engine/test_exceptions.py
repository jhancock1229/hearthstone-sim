"""
Unit tests for custom exceptions in hearthstone/exceptions.py.
"""

import pytest

from hearthstone import exceptions

def test_illegal_action_error_is_raised():
    with pytest.raises(exceptions.IllegalActionError):
        raise exceptions.IllegalActionError("Not allowed!")

def test_game_over_error_is_raised():
    with pytest.raises(exceptions.GameOverError):
        raise exceptions.GameOverError("Game is over!")

def test_illegal_action_error_inherits_from_exception():
    assert issubclass(exceptions.IllegalActionError, Exception)

def test_game_over_error_inherits_from_exception():
    assert issubclass(exceptions.GameOverError, Exception)

def test_illegal_action_error_message():
    err = exceptions.IllegalActionError("foo")
    assert str(err) == "foo"

def test_game_over_error_message():
    err = exceptions.GameOverError("bar")
    assert str(err) == "bar"
