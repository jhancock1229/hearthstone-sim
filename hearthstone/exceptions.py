
class IllegalActionError(Exception):
	"""Raised when an illegal action is attempted in the game."""
	pass

class GameOverError(Exception):
	"""Raised when an action is attempted after the game is over."""
	pass
