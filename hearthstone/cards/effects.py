"""Composable card effects: Battlecry, Deathrattle, Aura, Triggered.

Effects are attached to cards and fired by the event system. This module
provides registries for deathrattle and battlecry effects, supports scoping
battlecries to a `Game` instance, and utilities to load effects from
card data for simple data-driven effects.
"""

from typing import Any, Callable, Dict, Optional

from hearthstone.engine import events
from hearthstone.cards.base import MinionCard

# Deathrattle registry: name -> callable(player, minion)
_deathrattle_registry: Dict[str, Callable[[Any, Any], None]] = {}

# Battlecry registries:
# global: name -> callable(player, minion, **kwargs)
_battlecry_registry_global: Dict[str, Callable] = {}
# scoped: game -> { name -> callable }
_battlecry_registry_scoped: Dict[Any, Dict[str, Callable]] = {}

# Handlers stored so we can unregister them from the event bus
_battlecry_handlers_global: Dict[str, Callable] = {}
_battlecry_handlers_scoped: Dict[Any, Dict[str, Callable]] = {}


def register_deathrattle(name: str):
    def _dec(fn: Callable[[Any, Any], None]):
        _deathrattle_registry[name] = fn
        return fn

    return _dec


def trigger_deathrattle(player, minion):
    fn = _deathrattle_registry.get(minion.name)
    if fn:
        try:
            fn(player, minion)
        except Exception:
            pass


def unregister_deathrattle(name: str):
    _deathrattle_registry.pop(name, None)


def register_battlecry(name: str, game: Optional[Any] = None):
    """Register a battlecry handler for `name`.

    If `game` is provided the handler only triggers for events emitted with
    `game=<that game>` (scoped). The decorator registers an event listener
    on `events.bus` for `on_play_from_hand` that dispatches to the provided
    function when the minion name matches.
    """

    def _dec(fn: Callable[[Any, Any], None]):
        if game is None:
            _battlecry_registry_global[name] = fn

            def _handler(evt):
                minion = evt.data.get("minion")
                if not minion or getattr(minion, "name", None) != name:
                    return
                try:
                    _data = dict(evt.data)
                    _data.pop("minion", None)
                    fn(evt.source, minion, **_data)
                except Exception:
                    pass

            _battlecry_handlers_global[name] = _handler
            events.bus.on("on_play_from_hand", _handler)
        else:
            _battlecry_registry_scoped.setdefault(game, {})[name] = fn

            def _handler(evt):
                # Only trigger if event's game matches
                if evt.data.get("game") is not game:
                    return
                minion = evt.data.get("minion")
                if not minion or getattr(minion, "name", None) != name:
                    return
                try:
                    _data = dict(evt.data)
                    _data.pop("minion", None)
                    fn(evt.source, minion, **_data)
                except Exception:
                    pass

            _battlecry_handlers_scoped.setdefault(game, {})[name] = _handler
            events.bus.on("on_play_from_hand", _handler)

        return fn

    return _dec


def unregister_battlecry(name: str, game: Optional[Any] = None):
    if game is None:
        _battlecry_registry_global.pop(name, None)
        handler = _battlecry_handlers_global.pop(name, None)
        if handler:
            try:
                events.bus.off("on_play_from_hand", handler)
            except Exception:
                pass
    else:
        regs = _battlecry_registry_scoped.get(game, {})
        regs.pop(name, None)
        handlers = _battlecry_handlers_scoped.get(game, {})
        handler = handlers.pop(name, None)
        if handler:
            try:
                events.bus.off("on_play_from_hand", handler)
            except Exception:
                pass


def load_effects_from_card_data(card_data: Dict[str, Any]):
    """Load simple effects from a card data dict.

    Currently supports `battlecry` with action `summon` and a `token` spec.
    """
    name = card_data.get("name")
    effects_spec = card_data.get("effects", {})
    bc = effects_spec.get("battlecry")
    if not name or not bc:
        return

    action = bc.get("action")
    if action == "summon":
        token_spec = bc.get("token") or {}

        def _summon(player, minion, **kwargs):
            token = MinionCard(
                name=token_spec.get("name", "Token"),
                mana_cost=token_spec.get("mana_cost", 0),
                attack=token_spec.get("attack", 0),
                health=token_spec.get("health", 1),
            )
            try:
                player.place_minion(token)
            except Exception:
                pass

        register_battlecry(name)(_summon)


def clear_registries():
    # Clear deathrattle
    _deathrattle_registry.clear()

    # Unregister and clear battlecries global
    for name in list(_battlecry_handlers_global.keys()):
        unregister_battlecry(name)
    _battlecry_registry_global.clear()
    # Scoped handlers
    for game in list(_battlecry_handlers_scoped.keys()):
        for name in list(_battlecry_handlers_scoped.get(game, {}).keys()):
            unregister_battlecry(name, game=game)
    _battlecry_registry_scoped.clear()
    _battlecry_handlers_scoped.clear()
