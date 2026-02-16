"""Game zones: Deck, Hand, Board, Graveyard, Secrets.

Simple, test-focused implementations used by unit tests. Each zone
enforces basic constraints and provides a small API used by the test
suite (add/push/draw/remove/clear/shuffle, iteration, and len()).
"""

from __future__ import annotations

from typing import Iterable, Iterator, List, Any
import random


class Hand:
	MAX_SIZE = 10

	def __init__(self) -> None:
		self._cards: List[Any] = []

	def add(self, card: Any) -> None:
		if len(self._cards) >= self.MAX_SIZE:
			raise Exception("Hand is full")
		self._cards.append(card)

	def __len__(self) -> int:
		return len(self._cards)

	def __iter__(self) -> Iterator[Any]:
		return iter(self._cards)


class Board:
	BOARD_LIMIT = 7

	def __init__(self) -> None:
		self._minions: List[Any] = []

	def add(self, minion: Any, position: int | None = None) -> int:
		if len(self._minions) >= self.BOARD_LIMIT:
			raise Exception("Board is full (7 minions max)")
		if position is None:
			self._minions.append(minion)
			return len(self._minions) - 1
		if not (0 <= position <= len(self._minions)):
			raise ValueError("Invalid board position")
		self._minions.insert(position, minion)
		return position

	def remove(self, index: int) -> Any:
		if not (0 <= index < len(self._minions)):
			raise IndexError("No minion at index")
		return self._minions.pop(index)

	def __len__(self) -> int:
		return len(self._minions)

	def __iter__(self) -> Iterator[Any]:
		return iter(self._minions)


class Deck:
	def __init__(self) -> None:
		# simple stack semantics: push then draw (LIFO)
		self._cards: List[Any] = []

	def push(self, card: Any) -> None:
		self._cards.append(card)

	def draw(self) -> Any:
		if not self._cards:
			raise Exception("Deck is empty")
		return self._cards.pop()

	def shuffle(self) -> None:
		random.shuffle(self._cards)

	def __len__(self) -> int:
		return len(self._cards)

	def __iter__(self) -> Iterator[Any]:
		return iter(self._cards)


class Graveyard:
	def __init__(self) -> None:
		self._cards: List[Any] = []

	def add(self, card: Any) -> None:
		self._cards.append(card)

	def clear(self) -> None:
		self._cards.clear()

	def __len__(self) -> int:
		return len(self._cards)

	def __iter__(self) -> Iterator[Any]:
		return iter(self._cards)


class Secrets:
	MAX_SECRETS = 5

	def __init__(self) -> None:
		self._secrets: List[Any] = []

	def add(self, secret: Any) -> None:
		if len(self._secrets) >= self.MAX_SECRETS:
			raise Exception("Too many secrets")
		self._secrets.append(secret)

	def remove(self, secret: Any) -> None:
		try:
			self._secrets.remove(secret)
		except ValueError:
			raise

	def __len__(self) -> int:
		return len(self._secrets)

	def __iter__(self) -> Iterator[Any]:
		return iter(self._secrets)

