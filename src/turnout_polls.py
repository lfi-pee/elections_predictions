"""Published pre-scrutin participation polls ("indice de participation").

National turnout-intention measured BEFORE the vote, used as the national
abstention level exactly like vote-intention polls feed the vote blocks.
Only elections with a sourceable national estimate are listed; others fall
back to the historical estimator. Figures come from the same instrument
(Ipsos/CEVIPOF "Enquête électorale française", 0-10 certainty scale → %),
so they are comparable across the listed elections. Sourced & adversarially
verified 2026-06-06.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParticipationPoll:
    election_type: str
    date_float: float
    participation_pct: float
    source: str


PARTICIPATION_POLLS: list[ParticipationPoll] = [
    ParticipationPoll(
        "Legislatives_T1",
        2024.5,
        63.0,
        "Ipsos/CEVIPOF EEF vague 6, terrain 21-24 juin 2024",
    ),
    ParticipationPoll(
        "Legislatives_T1",
        2022.5,
        46.0,
        "Ipsos/CEVIPOF EEF vague 12, terrain 3-6 juin 2022",
    ),
    ParticipationPoll(
        "Presidentielle_T1",
        2022.33,
        67.0,
        "Ipsos/CEVIPOF EEF, ~1 mois avant le 1er tour 2022",
    ),
]


def national_abstention_from_poll(
    election_type: str, date_float: float
) -> tuple[float, str] | None:
    """Abstention (= 100 − participation) from a published participation poll,
    or None if no poll covers this election."""
    for p in PARTICIPATION_POLLS:
        if p.election_type == election_type and abs(p.date_float - date_float) < 0.3:
            return 100.0 - p.participation_pct, p.source
    return None
