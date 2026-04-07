"""
FastAPI server for the Indian Traffic Control OpenEnv environment.

Required endpoints:
  POST /reset     → Initialize episode, return initial observation
  POST /step      → Execute action, return StepResult
  GET  /state     → Return current episode state
  GET  /tasks     → Return task list and action schema
  POST /grader    → Grade a completed episode (0.0–1.0)
  POST /baseline  → Run heuristic baseline on all 3 tasks, return scores
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from models import TrafficAction, TrafficObservation, TrafficReward, TrafficState, StepResult
from server.environment import IndianTrafficEnvironment
from server.tasks import get_tasks, grade_episode, run_task_episode

# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Traffic Signal Control — OpenEnv",
    description=(
        "Autonomous Traffic Signal Control System for a 4-way intersection. "
        "Manages signal phases under red-light runners, potholes, weather, and emergency vehicles."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (one session per server process)
_env = IndianTrafficEnvironment(steps_per_episode=150)


# ─── Request / Response schemas ───────────────────────────────────────────────

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    task_id: Optional[str] = None
    options: Optional[Dict[str, Any]] = None


class GraderRequest(BaseModel):
    task_id: str
    episode_metrics: Dict[str, Any]


class BaselineResponse(BaseModel):
    scores: Dict[str, float]
    details: Dict[str, Any]


# ─── Standard OpenEnv Endpoints ───────────────────────────────────────────────

@app.post("/reset", response_model=TrafficObservation, summary="Reset the environment")
def reset(request: ResetRequest = ResetRequest()):
    """
    Initialize a new episode.
    Returns the initial observation.
    """
    obs = _env.reset(seed=request.seed, task_id=request.task_id)
    return obs


@app.post("/step", response_model=StepResult, summary="Execute one action")
def step(action: TrafficAction):
    """
    Execute one environment step.
    Returns observation, reward, done flag, and info dict.
    """
    try:
        result = _env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/state", response_model=TrafficState, summary="Get current episode state")
def state():
    """
    Return current episode metadata: episode_id, step_count, total_reward, done, crashed.
    """
    return _env.state()


# ─── Hackathon-Required Endpoints ─────────────────────────────────────────────

@app.get("/tasks", summary="List all tasks and action schema")
def tasks():
    """
    Returns the list of tasks (easy, medium, hard) and the full action schema.
    Required by the OpenEnv hackathon pre-submission checklist.
    """
    return {
        "tasks": get_tasks(),
        "action_schema": {
            "type": "object",
            "required": ["action"],
            "properties": {
                "action": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 5,
                    "description": (
                        "0=Keep Phase, 1=N/S Straight Green, 2=N/S Left Green, "
                        "3=E/W Straight Green, 4=E/W Left Green, 5=Pedestrian Walk"
                    ),
                }
            },
        },
        "observation_schema": {
            "type": "object",
            "properties": {
                "cars_ns_straight": {"type": "number", "description": "Cars queued N/S straight"},
                "cars_ns_left":     {"type": "number", "description": "Cars queued N/S left turn"},
                "cars_ew_straight": {"type": "number", "description": "Cars queued E/W straight"},
                "cars_ew_left":     {"type": "number", "description": "Cars queued E/W left turn"},
                "ambulance_ns":     {"type": "number", "description": "Ambulance on N/S (0 or 1)"},
                "ambulance_ew":     {"type": "number", "description": "Ambulance on E/W (0 or 1)"},
                "pedestrians_waiting": {"type": "number", "description": "Pedestrians waiting"},
                "current_phase":    {"type": "number", "description": "Signal phase index 0-9"},
                "time_in_phase":    {"type": "number", "description": "Steps in current phase"},
                "weather":          {"type": "number", "description": "0=Clear 1=Rain 2=Monsoon"},
                "pothole_ns":       {"type": "number", "description": "Pothole on N/S road (0/1)"},
                "pothole_ew":       {"type": "number", "description": "Pothole on E/W road (0/1)"},
            },
        },
    }


@app.post("/grader", summary="Grade a completed episode")
def grader(request: GraderRequest):
    """
    Grade a completed episode for the given task_id.
    Returns a deterministic score strictly in (0.0, 1.0) — exclusive endpoints.
    """
    try:
        score = grade_episode(request.task_id, request.episode_metrics)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "task_id": request.task_id,
        "score": score,
        "score_range": "(0.0, 1.0) exclusive",
    }


@app.post("/baseline", response_model=BaselineResponse, summary="Run baseline agent on all 3 tasks")
def baseline():
    """
    Runs the built-in heuristic baseline agent on all 3 tasks.
    Returns reproducible scores for easy, medium, and hard tasks.
    This endpoint does NOT require OPENAI_API_KEY.
    See baseline.py for the LLM-based baseline.
    """
    def heuristic_policy(obs):
        """
        Greedy heuristic: prioritise ambulances, then longest queue.
        Matches the logic in the original evaluator.py.
        """
        cars_ns_s, cars_ns_l, cars_ew_s, cars_ew_l = obs[0], obs[1], obs[2], obs[3]
        amb_ns, amb_ew = obs[4], obs[5]
        peds = obs[6]
        phase = int(obs[7])
        time_in_phase = obs[8]

        action = 0
        if time_in_phase >= 4:
            if amb_ns == 1 and phase != 0:
                action = 1
            elif amb_ew == 1 and phase != 4:
                action = 3
            elif peds > 4 and phase != 8:
                action = 5
            else:
                max_q = max(cars_ns_s, cars_ns_l, cars_ew_s, cars_ew_l)
                if max_q > 6:
                    if max_q == cars_ns_s and phase != 0:
                        action = 1
                    elif max_q == cars_ns_l and phase != 2:
                        action = 2
                    elif max_q == cars_ew_s and phase != 4:
                        action = 3
                    elif max_q == cars_ew_l and phase != 6:
                        action = 4
        return action

    scores = {}
    details = {}
    for task_id in ["easy", "medium", "hard"]:
        metrics = run_task_episode(task_id, heuristic_policy)
        score = grade_episode(task_id, metrics)
        scores[task_id] = round(score, 4)
        details[task_id] = {
            "steps_completed": metrics["steps_completed"],
            "crashed": metrics["crashed"],
            "mean_waiting": round(float(sum(metrics["total_waiting"]) / max(len(metrics["total_waiting"]), 1)), 2),
            "score": round(score, 4),
        }

    return BaselineResponse(scores=scores, details=details)


# ─── Health-check ─────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
def root():
    return {
        "environment": "IndianTrafficControl",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline"],
    }


def main():
    """Entry point for the server — required by openenv validate."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
