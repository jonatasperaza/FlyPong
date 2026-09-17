"""One fly = one genome, evaluated against the fixed Pong AI opponent.

This deliberately reuses `main.FlyPongRunner` unmodified (fly-vs-fixed-bot,
not fly-vs-fly): building a two-brain shared-game variant is a bigger change
to `game/pong.py` and `main.py` that belongs in a later phase, not bundled
into the first, independently-testable slice of the evolution work.

Honesty note on the genome-to-mechanism mapping: only `plastic_lr` and
`synthetic_w_init` are wired to something that actually changes simulated
behavior right now (`sim.network.PLASTIC_LR` and `main.SYNTHETIC_W_INIT`).
`dopamine_baseline`, `weight_scale`, `noise_sigma` and `action_threshold`
are part of the Genome dataclass (evolution/genome.py) but have no
corresponding mechanism in main.py/sim/network.py yet -- they are inert
until that wiring is added. Do not report a fitness difference as caused
by one of those fields; it cannot be, yet.
"""
from __future__ import annotations

from dataclasses import dataclass

import sim.network as netmod
from game.pong import PongGame
from evolution.genome import Genome

import main


@dataclass
class MatchStats:
    n_attempts: int
    bounce_rate_inicio: float | None
    bounce_rate_fim: float | None
    learning_gain: float | None  # bounce_rate_fim - bounce_rate_inicio, None if too few attempts


def _apply_genome(genome: Genome) -> None:
    """Set the module-level globals that main.py/sim.network read at call time.

    Same monkeypatch pattern used throughout this project's achados: these
    are plain module attributes, re-read on every step, so setting them
    before constructing a FlyPongRunner is enough -- no reload needed.
    """
    netmod.PLASTIC_LR = genome.plastic_lr
    main.SYNTHETIC_W_INIT = genome.synthetic_w_init


def play_one_life(
    genome: Genome,
    seed: int,
    frames: int,
    *,
    signal_mode: str = "real",
    plasticity_mode: str = "original",
) -> MatchStats:
    """Run one fly's lifetime against the fixed AI opponent and score it.

    signal_mode="real" is the default on purpose: it is the only mode that
    does not inject a forced steering current into DNa10, so a fitness
    measured here reflects the genome/circuit/plasticity, not a scaffold.
    Use signal_mode="synthetic_plastic" only when you explicitly want to
    measure "with the scaffold on" (e.g. for the genuine-learning ablation
    below, mirroring what the scaffold has always been used for since
    Achado 13).
    """
    netmod.PLASTICITY_MODE = plasticity_mode
    _apply_genome(genome)

    runner = main.FlyPongRunner(main.DATA_DIR, signal_mode=signal_mode)
    runner.game = PongGame(seed=seed)
    for _ in range(frames):
        runner.step_frame()

    report = runner.validation_report()
    if "bounce_rate_fim" not in report:
        return MatchStats(
            n_attempts=report["n_attempts"],
            bounce_rate_inicio=None,
            bounce_rate_fim=None,
            learning_gain=None,
        )
    return MatchStats(
        n_attempts=report["n_attempts"],
        bounce_rate_inicio=report["bounce_rate_inicio"],
        bounce_rate_fim=report["bounce_rate_fim"],
        learning_gain=report["delta"],
    )


def genuine_learning_gap(
    genome: Genome,
    seed: int,
    frames: int,
    *,
    signal_mode: str = "real",
) -> dict:
    """Ablation: same genome, same seed, plasticity on vs off.

    This is the check requested when the user asked whether a tournament
    winner would count as "the fly learned to play": a genome that wins
    with plastic_lr>0 but performs identically with plastic_lr forced to 0
    won a good *fixed* circuit, not a *learning* one. Report both runs so
    nobody downstream has to take a single fitness number on faith.
    """
    learning_on = play_one_life(genome, seed, frames, signal_mode=signal_mode)

    frozen_genome = Genome(**{**genome.to_dict(), "plastic_lr": 0.0})
    learning_off = play_one_life(frozen_genome, seed, frames, signal_mode=signal_mode)

    gap = None
    if learning_on.bounce_rate_fim is not None and learning_off.bounce_rate_fim is not None:
        gap = learning_on.bounce_rate_fim - learning_off.bounce_rate_fim

    return {
        "learning_on": learning_on,
        "learning_off": learning_off,
        "bounce_rate_fim_gap": gap,
        "verdict": (
            "dados insuficientes"
            if gap is None
            else "aprendizado genuino plausivel (plasticidade importa)"
            if gap > 0.05
            else "sem evidencia de aprendizado genuino (genoma fixo ja basta)"
        ),
    }
