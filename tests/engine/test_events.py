"""
Unit tests for the event system (TDD): EventBus and Event behavior.

Expected API (to be implemented in `hearthstone.engine.events`):
- `EventBus` class with methods `on(event_name, handler, *, once=False)`,
  `off(event_name, handler)`, and `emit(event_name, *, source=None, **data)`.
- `emit` should create an `Event` object passed to handlers with attributes
  `name`, `source`, `data` (dict), and `cancelled` (bool).
- `once=True` handlers should be removed after their first invocation.
- Setting `event.cancelled = True` in a handler should stop further handlers.
"""

import pytest
from hearthstone.engine import events


def test_register_and_emit_calls_handler():
    bus = events.EventBus()
    calls = []

    def handler(evt):
        calls.append(evt.data.get("value"))

    bus.on("test:event", handler)
    bus.emit("test:event", value=42)
    assert calls == [42]


def test_multiple_listeners_order():
    bus = events.EventBus()
    order = []

    def a(evt):
        order.append("a")

    def b(evt):
        order.append("b")

    bus.on("evt", a)
    bus.on("evt", b)
    bus.emit("evt")
    assert order == ["a", "b"]


def test_once_listener_removed_after_emit():
    bus = events.EventBus()
    count = {"n": 0}

    def handler(evt):
        count["n"] += 1

    bus.on("once:evt", handler, once=True)
    bus.emit("once:evt")
    bus.emit("once:evt")
    assert count["n"] == 1


def test_cancel_stops_propagation():
    bus = events.EventBus()
    called = []

    def stopper(evt):
        called.append("stopper")
        evt.cancelled = True

    def later(evt):
        called.append("later")

    bus.on("chain", stopper)
    bus.on("chain", later)
    bus.emit("chain")
    assert called == ["stopper"]


def test_off_unregisters_handler():
    bus = events.EventBus()
    called = []

    def h(evt):
        called.append(True)

    bus.on("x", h)
    bus.off("x", h)
    bus.emit("x")
    assert called == []


def test_event_contains_source_and_name():
    bus = events.EventBus()

    class S:
        pass

    src = S()

    def handler(evt):
        assert evt.name == "my:event"
        assert evt.source is src
        assert isinstance(evt.data, dict)

    bus.on("my:event", handler)
    bus.emit("my:event", source=src, foo="bar")


# Ensure the tests fail meaningfully if the module isn't implemented yet
def test_events_module_placeholder_detected():
    # The module currently should define EventBus and Event; if it doesn't,
    # this test asserts clearly so implementation can follow the spec.
    assert hasattr(events, "EventBus"), "EventBus must be implemented in events.py"
    assert hasattr(events, "Event"), "Event dataclass must be implemented in events.py"
