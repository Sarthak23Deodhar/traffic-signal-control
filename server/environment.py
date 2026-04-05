"""
Server-side OpenEnv Environment for the Indian Traffic Control environment.
Wraps the gymnasium-style IndianTrafficEnv with the OpenEnv interface.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
import numpy as np
from typing import Tuple, Dict, Any

from traffic_env import IndianTrafficEnv
from models import (
    TrafficAction,
    TrafficObservation,
    TrafficReward,
    TrafficState,
    StepResult,
)


class IndianTrafficEnvironment:
    """
    OpenEnv-compliant environment wrapping IndianTrafficEnv.

    Implements:
        reset()  -> TrafficObservation
        step()   -> StepResult
        state()  -> TrafficState
    """

    def __init__(self, steps_per_episode: int = 150):
        self._env = IndianTrafficEnv(steps_per_episode=steps_per_episode)
        self._state = TrafficState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            total_reward=0.0,
            done=False,
            crashed=False,
            task_id=None,
            max_steps=steps_per_episode,
        )
        self._last_obs: TrafficObservation | None = None

    # ─── OpenEnv API ─────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None, task_id: str | None = None) -> TrafficObservation:
        """Initialize a new episode. Returns the initial observation."""
        obs_arr, _ = self._env.reset(seed=seed)
        self._state = TrafficState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            total_reward=0.0,
            done=False,
            crashed=False,
            task_id=task_id,
            max_steps=self._env.steps_per_episode,
        )
        self._last_obs = self._arr_to_obs(obs_arr)
        return self._last_obs

    def step(self, action: TrafficAction) -> StepResult:
        """Execute one action. Returns observation, reward, done, info."""
        if self._last_obs is None:
            raise RuntimeError("Call reset() before step().")

        obs_arr, raw_reward, terminated, truncated, info = self._env.step(action.action)

        # Build named reward components from info
        components: Dict[str, float] = {
            "wait_penalty": -float(np.sum(self._env.cars)) * 1.5,
            "ped_penalty": -float(self._env.ped_waiting) * 3.0,
            "amb_ns_penalty": -100.0 if (self._env.amb_ns and self._env.phase != 0) else 0.0,
            "amb_ew_penalty": -100.0 if (self._env.amb_ew and self._env.phase != 4) else 0.0,
            "crash_penalty": -1000.0 if info.get("crashed") else 0.0,
        }

        reward = TrafficReward(value=float(raw_reward), components=components)
        done = terminated or truncated

        self._state.step_count += 1
        self._state.total_reward += float(raw_reward)
        self._state.done = done
        self._state.crashed = bool(info.get("crashed", False))

        self._last_obs = self._arr_to_obs(obs_arr)

        return StepResult(
            observation=self._last_obs,
            reward=reward,
            done=done,
            info={
                "total_waiting": float(info.get("total_waiting", 0.0)),
                "ped_waiting": float(info.get("ped_waiting", 0.0)),
                "phase": self._env.phase_map.get(int(self._env.phase), "unknown"),
                "crashed": bool(info.get("crashed", False)),
                "step": self._state.step_count,
            },
        )

    def state(self) -> TrafficState:
        """Return current episode metadata."""
        return self._state

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _arr_to_obs(self, arr: np.ndarray) -> TrafficObservation:
        return TrafficObservation(
            cars_ns_straight=float(arr[0]),
            cars_ns_left=float(arr[1]),
            cars_ew_straight=float(arr[2]),
            cars_ew_left=float(arr[3]),
            ambulance_ns=float(arr[4]),
            ambulance_ew=float(arr[5]),
            pedestrians_waiting=float(arr[6]),
            current_phase=float(arr[7]),
            time_in_phase=float(arr[8]),
            weather=float(arr[9]),
            pothole_ns=float(arr[10]),
            pothole_ew=float(arr[11]),
        )
