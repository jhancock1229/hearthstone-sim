"""Event system: on_damage, on_death, on_play, on_heal, on_summon, etc.

Provides a small `Event` dataclass and an `EventBus` for registering
and emitting events. This is a minimal, synchronous implementation
used by tests and simple triggered effects.

API:
- `Event(name, source=None, data={})` — event container passed to handlers.
- `EventBus.on(event_name, handler, *, once=False)` — register handler.
- `EventBus.off(event_name, handler)` — unregister handler.
- `EventBus.emit(event_name, *, source=None, **data)` — emit event.

Handlers receive the `Event` instance and may set `event.cancelled = True`
to stop further handlers. `once=True` handlers are removed after first call.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple


@dataclass
class Event:
	name: str
	source: Any = None
	data: Dict[str, Any] = field(default_factory=dict)
	cancelled: bool = False


class EventBus:
	"""Simple synchronous event bus.

	Listeners are stored in registration order. Each listener is a tuple of
	(callable, once_flag). Emitting an event will call listeners in order
	until all are called or `event.cancelled` is set to True.
	"""

	def __init__(self) -> None:
		self._listeners: Dict[str, List[Tuple[Callable[[Event], None], bool]]] = {}

	def on(self, event_name: str, handler: Callable[[Event], None], *, once: bool = False) -> None:
		self._listeners.setdefault(event_name, []).append((handler, once))

	def off(self, event_name: str, handler: Callable[[Event], None]) -> None:
		if event_name in self._listeners:
			self._listeners[event_name] = [t for t in self._listeners[event_name] if t[0] is not handler]
			if not self._listeners[event_name]:
				del self._listeners[event_name]

	def emit(self, event_name: str, *, source: Any = None, **data: Any) -> Event:
		listeners = list(self._listeners.get(event_name, []))
		evt = Event(name=event_name, source=source, data=data)
		# Expose data keys as attributes for convenience (e.g., evt.target)
		for k, v in data.items():
			try:
				setattr(evt, k, v)
			except Exception:
				pass
		for handler, once in listeners:
			handler(evt)
			if once:
				# remove the handler from the live registry
				try:
					self.off(event_name, handler)
				except Exception:
					pass
			if evt.cancelled:
				break
		return evt


# Convenience global bus for the simulator
bus = EventBus()

