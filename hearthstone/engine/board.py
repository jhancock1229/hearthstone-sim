
"""
Board state: manages minion slots and board operations for a player.
"""

class Board:
	BOARD_LIMIT = 7

	def __init__(self):
		self.minions = []

	def add_minion(self, minion, position=None):
		if len(self.minions) >= self.BOARD_LIMIT:
			raise Exception("Board is full (7 minions max)")
		if position is None:
			self.minions.append(minion)
			return len(self.minions) - 1
		if not (0 <= position <= len(self.minions)):
			raise ValueError(f"Invalid board position: {position}")
		self.minions.insert(position, minion)
		return position

	def remove_minion(self, index):
		if not (0 <= index < len(self.minions)):
			raise IndexError(f"No minion at index {index}")
		return self.minions.pop(index)

	def clear(self):
		self.minions.clear()

	def __len__(self):
		return len(self.minions)

	def __iter__(self):
		return iter(self.minions)

	def __str__(self):
		return f"Board({[m.name for m in self.minions]})"

	def __repr__(self):
		return f"Board({[repr(m) for m in self.minions]})"
