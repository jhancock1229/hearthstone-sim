"""
Unit tests for additional event names we expect to support in the engine.
These tests assert the `EventBus` can register and emit the following events
and that handlers receive the Event with name and data.

This is a TDD step; later we'll wire real engine actions to emit these events.
"""

from hearthstone.engine import events


EVENT_NAMES = [
    "on_play_from_hand",
    "on_deathrattle",
    "on_turn_start",
    "on_turn_end",
    "on_attack",
    "on_spell_cast",
    "on_spell_target",
    "on_minion_summoned",
    "on_armor_change",
    "on_secret_trigger",
]


def test_event_names_emit_and_handlers_receive_data():
    bus = events.EventBus()
    called = []

    for name in EVENT_NAMES:
        def make_handler(n):
            def handler(evt):
                # record event name and sample payload
                called.append((n, evt.name, evt.data.get("payload")))
            return handler

        handler = make_handler(name)
        bus.on(name, handler)
        bus.emit(name, payload={"x": 1})

    # Each event should have been handled exactly once in order
    assert len(called) == len(EVENT_NAMES)
    for idx, (expected_name, evt_name, payload) in enumerate(called):
        assert evt_name == EVENT_NAMES[idx]
        assert payload == {"x": 1}
