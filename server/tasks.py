"""
Three tasks for the Indian Traffic Control environment.
Each task has:
  - A deterministic setup (fixed seed)
  - A run_episode() function that simulates an agent policy
  - A grade() function returning a float in [0.0, 1.0]
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Callable, Dict, Any

from traffic_env import IndianTrafficEnv


# ─── Task Registry ────────────────────────────────────────────────────────────

TASKS = [
    {
        "id": "easy",
        "name": "Clear the Intersection",
        "difficulty": "easy",
        "description": (
            "Minimize total vehicle wait time over 50 steps. "
            "Starts with clear weather, low traffic, no ambulances. "
            "Score = 1.0 - (mean_waiting / max_possible_waiting). "
            "A perfect agent scores 1.0; doing nothing scores ~0.0."
        ),
        "max_steps": 50,
        "seed": 42,
        "action_schema": {
            "action": {
                "type": "integer",
                "minimum": 0,
                "maximum": 5,
                "description": "Traffic signal phase action (0-5)"
            }
        },
    },
    {
        "id": "medium",
        "name": "Emergency Vehicle Handler",
        "difficulty": "medium",
        "description": (
            "An ambulance spawns in step 10. Clear it within 15 steps without a crash. "
            "Survive 80 steps total. "
            "Score = crash_free_bonus * (1 - amb_delay_fraction) * survival_fraction."
        ),
        "max_steps": 80,
        "seed": 123,
        "action_schema": {
            "action": {
                "type": "integer",
                "minimum": 0,
                "maximum": 5,
                "description": "Traffic signal phase action (0-5)"
            }
        },
    },
    {
        "id": "hard",
        "name": "Monsoon Crisis Manager",
        "difficulty": "hard",
        "description": (
            "Monsoon weather (weather=2), potholes on both roads, multiple ambulances. "
            "Survive 150 steps with zero crashes. "
            "Score = (1-crash_penalty) * ambulance_score * pedestrian_score."
        ),
        "max_steps": 150,
        "seed": 999,
        "action_schema": {
            "action": {
                "type": "integer",
                "minimum": 0,
                "maximum": 5,
                "description": "Traffic signal phase action (0-5)"
            }
        },
    },
]


def get_tasks():
    """Return the list of task descriptors."""
    return TASKS


def get_task_by_id(task_id: str) -> Dict[str, Any]:
    for t in TASKS:
        if t["id"] == task_id:
            return t
    raise ValueError(f"Unknown task_id: {task_id!r}")


# ─── Graders ──────────────────────────────────────────────────────────────────

def grade_episode(task_id: str, episode_metrics: Dict[str, Any]) -> float:
    """
    Score a completed episode for the given task.

    Args:
        task_id:          one of 'easy', 'medium', 'hard'
        episode_metrics:  dict collected during the episode:
            - steps_completed  (int)
            - crashed          (bool)
            - total_waiting    (List[float])  – per-step total cars waiting
            - amb_delays       (List[int])    – steps each ambulance waited
            - ped_clearance    (float)        – fraction of pedestrians cleared

    Returns:
        score in [0.0, 1.0]
    """
    if task_id == "easy":
        return _grade_easy(episode_metrics)
    elif task_id == "medium":
        return _grade_medium(episode_metrics)
    elif task_id == "hard":
        return _grade_hard(episode_metrics)
    else:
        raise ValueError(f"Unknown task_id: {task_id!r}")


def _grade_easy(m: Dict[str, Any]) -> float:
    """
    Score = 1 - (mean_wait / max_possible_wait).
    max_possible_wait = 50 cars (max_cars=50) per step.
    Crash incurs a 20% penalty multiplier.
    Clipped to [0, 1].
    """
    if not m.get("total_waiting"):
        return 0.0
    mean_wait = float(np.mean(m["total_waiting"]))
    max_possible = 50.0
    score = 1.0 - (mean_wait / max_possible)
    if m.get("crashed"):
        score *= 0.8   # 20% crash penalty but not instant zero
    return float(np.clip(score, 0.0, 1.0))


def _grade_medium(m: Dict[str, Any]) -> float:
    """
    crash_free_bonus  = 1.0 if no crash else 0.2
    amb_delay_score   = max(0, 1 - total_amb_delay / 20.0)
    survival_fraction = steps_completed / 80
    Score = crash_free * amb_delay_score * survival_fraction
    """
    crashed = bool(m.get("crashed", False))
    crash_free = 1.0 if not crashed else 0.2

    total_amb_delay = sum(m.get("amb_delays", [0]))
    amb_score = max(0.0, 1.0 - total_amb_delay / 20.0)

    steps = int(m.get("steps_completed", 0))
    survival = steps / 80.0

    score = crash_free * amb_score * survival
    return float(np.clip(score, 0.0, 1.0))


def _grade_hard(m: Dict[str, Any]) -> float:
    """
    crash_penalty   = 0.4 if crashed (60% reduction), 1.0 if crash-free
    amb_score       = max(0, 1 - mean_amb_delay / 10.0)
    ped_score       = ped_clearance fraction (0..1)
    congestion_score = 1 - mean_wait / 50
    Score = crash_factor * (0.4*amb_score + 0.3*ped_score + 0.3*congestion_score)
    A crash-free run can achieve ~0.6–0.9; a run with crashes is penalised to ~0.15–0.4.
    """
    crashed = bool(m.get("crashed", False))
    crash_factor = 0.4 if crashed else 1.0  # heavy but not zero

    delays = m.get("amb_delays", [0])
    mean_delay = float(np.mean(delays)) if delays else 0.0
    amb_score = float(np.clip(1.0 - mean_delay / 10.0, 0.0, 1.0))

    ped_score = float(np.clip(m.get("ped_clearance", 0.0), 0.0, 1.0))

    mean_wait = float(np.mean(m["total_waiting"])) if m.get("total_waiting") else 50.0
    congestion_score = float(np.clip(1.0 - mean_wait / 50.0, 0.0, 1.0))

    score = crash_factor * (0.4 * amb_score + 0.3 * ped_score + 0.3 * congestion_score)
    return float(np.clip(score, 0.0, 1.0))


# ─── Episode runner (used by /baseline endpoint) ──────────────────────────────

def run_task_episode(task_id: str, policy_fn: Callable[[np.ndarray], int]) -> Dict[str, Any]:
    """
    Run a full episode for the given task using the provided policy function.

    Args:
        task_id:   'easy' | 'medium' | 'hard'
        policy_fn: callable(obs_array) -> action (int 0-5)

    Returns:
        episode_metrics dict suitable for grade_episode()
    """
    task = get_task_by_id(task_id)
    max_steps = task["max_steps"]
    seed = task["seed"]

    # Fix global numpy seed for full reproducibility
    np.random.seed(seed)

    env = IndianTrafficEnv(steps_per_episode=max_steps)
    obs, _ = env.reset(seed=seed)

    # Easy: ensure clear weather and no potholes for a fair baseline
    if task_id == "easy":
        env.weather = 0
        env.pothole_ns = 0
        env.pothole_ew = 0

    # Medium: force an ambulance at step 10
    force_amb_step = 10 if task_id == "medium" else None
    if task_id == "medium":
        env.weather = 0

    # Hard: force monsoon + potholes (the whole challenge)
    if task_id == "hard":
        env.weather = 2
        env.pothole_ns = 1
        env.pothole_ew = 1

    total_waiting = []
    amb_delays = []
    current_amb_delay = 0
    crashed = False
    steps_done = 0

    for step in range(max_steps):
        if force_amb_step is not None and step == force_amb_step:
            env.amb_ns = 1

        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        steps_done = step + 1

        total_waiting.append(float(info.get("total_waiting", 0.0)))

        # Track ambulance delays
        amb_waiting = bool(env.amb_ns) or bool(env.amb_ew)
        if amb_waiting:
            current_amb_delay = int(current_amb_delay) + 1
        else:
            if current_amb_delay > 0:
                amb_delays.append(int(current_amb_delay))
                current_amb_delay = 0

        if info.get("crashed"):
            crashed = True
            # Reset crash state so the episode continues running all max_steps.
            # The crash is already recorded; graders penalise it in the score.
            env.crashed = False

        if truncated:
            break

    if current_amb_delay > 0:
        amb_delays.append(current_amb_delay)

    # Pedestrian clearance: fraction of steps with 0 pedestrians waiting
    ped_cleared_steps = sum(1 for w in total_waiting if w == 0)
    ped_clearance = ped_cleared_steps / max(steps_done, 1)

    return {
        "task_id": task_id,
        "steps_completed": steps_done,
        "crashed": crashed,
        "total_waiting": total_waiting,
        "amb_delays": amb_delays if amb_delays else [0],
        "ped_clearance": ped_clearance,
    }
