"""Simple genetic algorithm for evolving decks using the simulator.

This is a lightweight, easily-tunable implementation intended for
experimentation. It evaluates decks by simulating short matches between
two players using the minimalist play policy in `play_match`.
"""
import random
import statistics
from multiprocessing import Pool
from typing import List, Tuple

from deckbuilding.deck import random_deck, build_concrete_deck, CARD_POOL
from hearthstone.engine.player import Player
from hearthstone.engine.combat import resolve_minion_attack_hero, resolve_minion_attack


def play_match(genotype_a: List[str], genotype_b: List[str], max_turns: int = 50) -> int:
    """Play a simplified match between two decks.

    Returns 1 if A wins, 0 if B wins, 0.5 for draw.
    This is intentionally simple: each turn each player draws, plays the
    first card in hand (if any) to the board, then all board minions attack
    the enemy hero in order.
    """
    a = Player()
    b = Player()
    a.deck = build_concrete_deck(genotype_a)
    b.deck = build_concrete_deck(genotype_b)

    for turn in range(max_turns):
        # A's turn: draw and play
        a.draw_card()
        if a.hand:
            try:
                card = a.hand.pop(0)
                a.place_minion(card)
            except Exception:
                pass
        # A attacks with all minions to hero
        for attacker in list(a.board):
            if attacker in a.board:
                idx = a.board.index(attacker)
                try:
                    resolve_minion_attack_hero(a, idx, b)
                except Exception:
                    pass
        if b.is_dead:
            return 1

        # B's turn
        b.draw_card()
        if b.hand:
            try:
                card = b.hand.pop(0)
                b.place_minion(card)
            except Exception:
                pass
        for attacker in list(b.board):
            if attacker in b.board:
                idx = b.board.index(attacker)
                try:
                    resolve_minion_attack_hero(b, idx, a)
                except Exception:
                    pass
        if a.is_dead:
            return 0

    # If neither died, compare health as tiebreaker
    if a.health > b.health:
        return 1
    if b.health > a.health:
        return 0
    return 0.5


def evaluate_deck(genotype: List[str], opponent_genotype: List[str], games: int = 20) -> float:
    wins = 0
    for _ in range(games):
        res = play_match(genotype, opponent_genotype)
        if res == 1:
            wins += 1
        elif res == 0.5:
            wins += 0.5
    return wins / games


def tournament(population: List[List[str]], opponent: List[str], games_per_eval: int = 20, processes: int = 1) -> List[Tuple[List[str], float]]:
    eval_args = [(ind, opponent, games_per_eval) for ind in population]
    if processes > 1:
        with Pool(processes) as p:
            scores = p.starmap(evaluate_deck, eval_args)
    else:
        scores = [evaluate_deck(*args) for args in eval_args]
    return list(zip(population, scores))


def crossover(a: List[str], b: List[str]) -> Tuple[List[str], List[str]]:
    if len(a) != len(b):
        raise ValueError("Genotypes must be same length")
    n = len(a)
    i = random.randint(1, n - 2)
    j = random.randint(i + 1, n - 1)
    ca = a[:i] + b[i:j] + a[j:]
    cb = b[:i] + a[i:j] + b[j:]
    return ca, cb


def mutate(genotype: List[str], pool_keys: List[str], rate: float = 0.05) -> List[str]:
    out = genotype[:]
    for i in range(len(out)):
        if random.random() < rate:
            out[i] = random.choice(pool_keys)
    return out


def run_ga(pop_size: int = 50, generations: int = 10, deck_size: int = 30, opponent: List[str] | None = None, games_per_eval: int = 20, processes: int = 1):
    pool_keys = list(CARD_POOL.keys())
    # initialize population
    population = [random_deck(CARD_POOL, deck_size) for _ in range(pop_size)]
    if opponent is None:
        # baseline opponent: random deck
        opponent = random_deck(CARD_POOL, deck_size)

    for gen in range(generations):
        scored = tournament(population, opponent, games_per_eval, processes)
        scored.sort(key=lambda x: x[1], reverse=True)
        best_score = scored[0][1]
        print(f"Gen {gen}: best={best_score:.3f}")

        # selection: keep top 10% as elites
        elites = [ind for ind, s in scored[: max(1, pop_size // 10)]]

        # build next generation
        next_pop = elites[:]
        while len(next_pop) < pop_size:
            parents = random.choices(scored, k=2)
            p1 = parents[0][0]
            p2 = parents[1][0]
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1, pool_keys)
            c2 = mutate(c2, pool_keys)
            next_pop.append(c1)
            if len(next_pop) < pop_size:
                next_pop.append(c2)

        population = next_pop

    # Final scoring
    final = tournament(population, opponent, games_per_eval, processes)
    final.sort(key=lambda x: x[1], reverse=True)
    return final
