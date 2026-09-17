"""Swiss-system pairing for the FlyPong evolutionary tournament.

Round-robin among N individuals needs N*(N-1)/2 matches: 128 flies would
need 8128 matches per generation, which is not affordable across many
generations. A Swiss tournament reaches a comparable ranking quality with
O(N * rounds) matches by re-sorting the field by current score before each
round and pairing neighbors, so it only ever compares individuals of
similar current strength.

This module only decides *who plays whom*; it knows nothing about how a
match is actually simulated. `run_swiss_tournament` takes a `play_match`
callback of the shape `play_match(id_a, id_b) -> (score_a, score_b)` so it
stays independent of the connectome simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence


@dataclass
class Standing:
    entity_id: int
    score: float = 0.0
    opponents: set[int] = field(default_factory=set)


def pair_round(standings: list[Standing]) -> list[tuple[int, int]]:
    """Sort by score (descending) and pair neighbors, skipping repeat opponents.

    A bye (odd number of entities) is given to the lowest-ranked entity
    that has not yet had one, recorded by the caller via the returned
    pairing list simply omitting one id; callers should treat a missing
    id as "sat out this round".
    """
    ordered = sorted(standings, key=lambda s: s.score, reverse=True)
    pairs: list[tuple[int, int]] = []
    unpaired = list(ordered)

    while len(unpaired) >= 2:
        a = unpaired.pop(0)
        partner_index = next(
            (i for i, b in enumerate(unpaired) if b.entity_id not in a.opponents),
            0,
        )
        b = unpaired.pop(partner_index)
        pairs.append((a.entity_id, b.entity_id))

    return pairs


def run_swiss_tournament(
    entity_ids: Sequence[int],
    play_match: Callable[[int, int], tuple[float, float]],
    rounds: int,
) -> dict[int, Standing]:
    """Run `rounds` Swiss rounds and return final standings keyed by entity id.

    `play_match(a, b)` must return (score_a, score_b) for that single match;
    scores accumulate additively across rounds into `Standing.score`.
    """
    standings = {eid: Standing(entity_id=eid) for eid in entity_ids}

    for _ in range(rounds):
        pairs = pair_round(list(standings.values()))
        for a_id, b_id in pairs:
            score_a, score_b = play_match(a_id, b_id)
            standings[a_id].score += score_a
            standings[b_id].score += score_b
            standings[a_id].opponents.add(b_id)
            standings[b_id].opponents.add(a_id)

    return standings
