"""Fly-vs-fly duel: two independent ConnectomeNetwork brains sharing one
DuelPongGame, each with its own genome-derived learning rate.

Deliberately does not reuse main.FlyPongRunner: that class is built around
one brain vs. the fixed AI bot (calibration, single game instance, single
set of reward/synthetic-current buffers) and is left untouched for Achado
18 and the rest of the project. This module only imports read-only pieces
that were always meant to be reused (ConnectomeNetwork, the sensory/motor
helper functions) plus a few constants from main.py, and adds its own
two-brain stepping loop on top.

signal_mode is intentionally fixed to "real" here (no synthetic push-pull
scaffold into DNa10): a duel is exactly the setting where we want to know
whether the circuit + its own plasticity can play, per the ablation
discussed with the user, not whether a forced steering current can.

Global-state gotcha: sim.network.ConnectomeNetwork.step() reads
PLASTIC_LR/PLASTICITY_MODE from the sim.network module namespace, not from
an instance attribute, so two brains with two different learning rates
cannot literally step "at the same time" -- this loop sets the global
immediately before each brain's own .step() call and restores it before
the other's. This is safe only because Python here is single-threaded and
the two calls are sequential within one frame; do not parallelize the two
brains inside a single duel with threads without revisiting this.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import sim.network as netmod
from sim.network import ConnectomeNetwork
from game.sensory_map import ball_to_photoreceptor_stimulus
from game.motor_read import read_motor_action

import main
from evolution.duel_pong import DuelPongGame
from evolution.genome import Genome

SUBSTEPS_PER_FRAME = main.SUBSTEPS_PER_FRAME
REWARD_PULSE_STEPS = main.REWARD_PULSE_STEPS
REWARD_CURRENT = main.REWARD_CURRENT
CALIBRATION_FRAMES = main.CALIBRATION_FRAMES
BOUNCE_RATE_WINDOW = main.BOUNCE_RATE_WINDOW


@dataclass
class _BrainState:
    net: ConnectomeNetwork
    plastic_lr: float
    motor_spike_accum: np.ndarray
    external: np.ndarray
    reward_pulse_remaining: int = 0
    reward_valence: float = 0.0
    motor_baseline_diff: float = 0.0
    point_outcomes: list | None = None
    bounce_rate_curve: list | None = None

    def __post_init__(self):
        self.point_outcomes = []
        self.bounce_rate_curve = []


def _calibrate(state: _BrainState, frames: int) -> float:
    stim = ball_to_photoreceptor_stimulus(
        main.HEIGHT / 2, main.WIDTH / 2, main.HEIGHT, main.WIDTH,
        state.net.photoreceptor_positions,
    )
    diffs = []
    netmod.PLASTIC_LR = 0.0  # never learn from the centered calibration stimulus
    for _ in range(frames):
        state.motor_spike_accum.fill(0.0)
        for _ in range(SUBSTEPS_PER_FRAME):
            state.external.fill(0.0)
            state.external[state.net.photoreceptor_idx] += stim
            state.motor_spike_accum += state.net.step(state.external)
        up = (
            state.motor_spike_accum[state.net.motor_up_idx].mean()
            if len(state.net.motor_up_idx) else 0.0
        )
        down = (
            state.motor_spike_accum[state.net.motor_down_idx].mean()
            if len(state.net.motor_down_idx) else 0.0
        )
        diffs.append(down - up)
    state.net.reset()
    return float(np.mean(diffs))


_network_cache: dict[str, ConnectomeNetwork] = {}

def _get_network() -> ConnectomeNetwork:
    key = str(main.DATA_DIR)
    if key not in _network_cache:
        _network_cache[key] = ConnectomeNetwork(data_dir=main.DATA_DIR)
    net = _network_cache[key]
    net.reset()
    return net

def _make_brain(genome: Genome, calibration_frames: int) -> _BrainState:
    net = _get_network()
    state = _BrainState(
        net=net,
        plastic_lr=genome.plastic_lr,
        motor_spike_accum=np.zeros(net.n, dtype=np.float64),
        external=np.zeros(net.n, dtype=np.float64),
    )
    state.motor_baseline_diff = _calibrate(state, calibration_frames)
    return state


def _reward_current(state: _BrainState) -> np.ndarray:
    state.external.fill(0.0)
    if state.reward_pulse_remaining > 0:
        idx = state.net.dopamine_positive_idx if state.reward_valence > 0 else state.net.dopamine_negative_idx
        state.external[idx] = REWARD_CURRENT
        state.reward_pulse_remaining -= 1
    return state.external


def _step_brain(state: _BrainState, ball_y: float, ball_x: float, mirror_x: bool) -> int:
    """Run one frame for one brain; returns its action (-1/0/1)."""
    state.net.reset()
    x = (main.WIDTH - ball_x) if mirror_x else ball_x
    stim = ball_to_photoreceptor_stimulus(
        ball_y, x, main.HEIGHT, main.WIDTH, state.net.photoreceptor_positions,
    )
    state.motor_spike_accum.fill(0.0)
    netmod.PLASTIC_LR = state.plastic_lr
    for _ in range(SUBSTEPS_PER_FRAME):
        external = _reward_current(state)
        external[state.net.photoreceptor_idx] += stim
        spikes = state.net.step(external)
        state.motor_spike_accum += spikes
    return read_motor_action(
        state.motor_spike_accum, state.net.motor_up_idx, state.net.motor_down_idx,
        baseline_diff=state.motor_baseline_diff,
    )


def _record_outcome(state: _BrainState, bounced: bool, missed: bool) -> None:
    if bounced:
        state.point_outcomes.append("bounce")
        state.reward_valence = 1.0
        state.reward_pulse_remaining = REWARD_PULSE_STEPS
    elif missed:
        state.point_outcomes.append("miss")
        state.reward_valence = -1.0
        state.reward_pulse_remaining = REWARD_PULSE_STEPS
    if bounced or missed:
        recent = state.point_outcomes[-BOUNCE_RATE_WINDOW:]
        state.bounce_rate_curve.append(recent.count("bounce") / len(recent))


def _max_rally(outcomes: list) -> int:
    best = current = 0
    for o in outcomes:
        current = current + 1 if o == "bounce" else 0
        best = max(best, current)
    return best


@dataclass
class SideStats:
    score: int
    n_bounces: int
    max_rally: int
    bounce_rate_inicio: float | None
    bounce_rate_fim: float | None
    learning_gain: float | None  # bounce_rate_fim - bounce_rate_inicio


@dataclass
class DuelResult:
    left: SideStats
    right: SideStats


def make_duel(genome_left: Genome, genome_right: Genome, seed: int, calibration_frames: int = CALIBRATION_FRAMES):
    """Build the (left_brain, right_brain, game) triple for a fresh duel.

    Exposed separately from play_duel so a caller that wants to render each
    frame (e.g. a pygame window) can drive step_duel_frame itself instead of
    running the whole match headless in one call.
    """
    left = _make_brain(genome_left, calibration_frames)
    right = _make_brain(genome_right, calibration_frames)
    game = DuelPongGame(seed=seed)
    return left, right, game


def step_duel_frame(left: _BrainState, right: _BrainState, game: DuelPongGame) -> dict:
    """Advance one frame: both brains act, physics steps, outcomes recorded.

    Returns the raw event dict from DuelPongGame.step (bounce_left,
    bounce_right, score) so a caller can react to it (e.g. flash the screen
    on a point) without having to re-derive it.
    """
    action_left = _step_brain(left, game.ball_y, game.ball_x, mirror_x=False)
    action_right = _step_brain(right, game.ball_y, game.ball_x, mirror_x=True)
    event = game.step(action_left, action_right)

    left_missed = event["score"] == "right"
    right_missed = event["score"] == "left"
    _record_outcome(left, event["bounce_left"], left_missed)
    _record_outcome(right, event["bounce_right"], right_missed)
    return event


def play_duel(
    genome_left: Genome,
    genome_right: Genome,
    seed: int,
    frames: int,
    calibration_frames: int = CALIBRATION_FRAMES,
    target_score: int | None = None,
) -> DuelResult:
    """frames is the match length when target_score is None (original
    behavior). When target_score is set, `frames` instead becomes a safety
    cap: the duel ends as soon as either side reaches target_score, OR after
    `frames` engine steps, whichever comes first. The cap matters because
    early-generation genomes can be too weak to ever reach target_score --
    without it, a single bad matchup could run forever."""
    left, right, game = make_duel(genome_left, genome_right, seed, calibration_frames)

    for _ in range(frames):
        step_duel_frame(left, right, game)
        if target_score is not None and (game.score_left >= target_score or game.score_right >= target_score):
            break

    def _rate(curve: list, head: bool) -> float | None:
        if len(curve) < 4:
            return None
        k = max(1, len(curve) // 5)
        return float(np.mean(curve[:k])) if head else float(np.mean(curve[-k:]))

    def _side_stats(state: _BrainState, score: int) -> SideStats:
        inicio = _rate(state.bounce_rate_curve, head=True)
        fim = _rate(state.bounce_rate_curve, head=False)
        gain = (fim - inicio) if (inicio is not None and fim is not None) else None
        return SideStats(
            score=score,
            n_bounces=state.point_outcomes.count("bounce"),
            max_rally=_max_rally(state.point_outcomes),
            bounce_rate_inicio=inicio,
            bounce_rate_fim=fim,
            learning_gain=gain,
        )

    return DuelResult(
        left=_side_stats(left, game.score_left),
        right=_side_stats(right, game.score_right),
    )
