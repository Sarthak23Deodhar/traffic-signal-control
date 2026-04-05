"""
Inference Script — Indian Traffic Control OpenEnv
=================================================
MANDATORY env vars:
    API_BASE_URL       The LLM API endpoint          (default: HuggingFace router)
    MODEL_NAME         The model identifier          (default: Qwen2.5-72B-Instruct)
    HF_TOKEN           Your HuggingFace / API key   (no default)
    LOCAL_IMAGE_NAME   Only if using from_docker_image() — not used here

STDOUT FORMAT (must match exactly):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
from typing import List, Optional

import numpy as np
import requests
from openai import OpenAI

# ── Mandatory env vars (pattern must match sample exactly) ─────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")                   # NO default
# LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")     # only needed if using from_docker_image()

API_KEY      = HF_TOKEN or os.getenv("OPENAI_API_KEY")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860").rstrip("/")

BENCHMARK    = "indian-traffic-control"
SUCCESS_SCORE_THRESHOLD = 0.1

# Task configs — seeds/max_steps must match openenv.yaml exactly
TASKS = [
    {"id": "easy",   "name": "clear-intersection",       "seed": 42,  "max_steps": 50},
    {"id": "medium", "name": "emergency-vehicle-handler", "seed": 123, "max_steps": 80},
    {"id": "hard",   "name": "monsoon-crisis-manager",   "seed": 999, "max_steps": 150},
]


# ── Structured log helpers (format must match sample exactly) ──────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Observation helpers ────────────────────────────────────────────────────────

def obs_dict_to_array(obs: dict) -> np.ndarray:
    return np.array([
        obs["cars_ns_straight"], obs["cars_ns_left"],
        obs["cars_ew_straight"], obs["cars_ew_left"],
        obs["ambulance_ns"],     obs["ambulance_ew"],
        obs["pedestrians_waiting"],
        obs["current_phase"],    obs["time_in_phase"],
        obs["weather"],
        obs["pothole_ns"],       obs["pothole_ew"],
    ], dtype=np.float32)


def build_obs_prompt(obs: np.ndarray) -> str:
    weather_names = ["Clear", "Rain", "Monsoon"]
    return (
        f"Intersection state:\n"
        f"  Cars: N/S straight={obs[0]:.0f}, N/S left={obs[1]:.0f}, "
        f"E/W straight={obs[2]:.0f}, E/W left={obs[3]:.0f}\n"
        f"  Ambulance: N/S={'YES' if obs[4] else 'no'}, E/W={'YES' if obs[5] else 'no'}\n"
        f"  Pedestrians waiting: {obs[6]:.0f}\n"
        f"  Signal phase: {int(obs[7])} (time in phase: {obs[8]:.0f})\n"
        f"  Weather: {weather_names[int(obs[9])]}, "
        f"Potholes: N/S={'yes' if obs[10] else 'no'} E/W={'yes' if obs[11] else 'no'}\n\n"
        f"Actions:\n"
        f"  0=Keep current phase  1=N/S Straight Green  2=N/S Left-Turn Green\n"
        f"  3=E/W Straight Green  4=E/W Left-Turn Green 5=All-Way Pedestrian Walk\n"
        f"Reply with ONE integer (0-5) only."
    )


# ── Fallback heuristic ─────────────────────────────────────────────────────────

def heuristic_action(obs: np.ndarray) -> int:
    cars_ns_s, cars_ns_l, cars_ew_s, cars_ew_l = obs[0], obs[1], obs[2], obs[3]
    amb_ns, amb_ew, peds = obs[4], obs[5], obs[6]
    phase, time_in_phase = int(obs[7]), obs[8]

    if time_in_phase < 4:
        return 0
    if amb_ns == 1 and phase != 0:
        return 1
    if amb_ew == 1 and phase != 4:
        return 3
    if peds > 4 and phase != 8:
        return 5
    max_q = max(cars_ns_s, cars_ns_l, cars_ew_s, cars_ew_l)
    if max_q > 6:
        if max_q == cars_ns_s and phase != 0: return 1
        if max_q == cars_ns_l and phase != 2: return 2
        if max_q == cars_ew_s and phase != 4: return 3
        if max_q == cars_ew_l and phase != 6: return 4
    return 0


# ── LLM action — uses OpenAI Client with API_BASE_URL / MODEL_NAME / HF_TOKEN ──

def get_llm_action(client: OpenAI, obs: np.ndarray) -> tuple[int, str]:
    prompt = build_obs_prompt(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": (
                    "You are an expert traffic signal controller. "
                    "Minimize congestion, prioritize emergency vehicles, protect pedestrians. "
                    "Reply with exactly one digit 0-5."
                )},
                {"role": "user", "content": prompt},
            ],
            max_tokens=5,
            temperature=0.0,
        )
        text = (completion.choices[0].message.content or "").strip()
        action = int(text[0])
        if 0 <= action <= 5:
            return action, str(action)
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)

    fallback = heuristic_action(obs)
    return fallback, f"heuristic({fallback})"


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_task(client: Optional[OpenAI], task: dict) -> None:
    task_id   = task["id"]
    task_name = task["name"]
    seed      = task["seed"]
    max_steps = task["max_steps"]

    rewards:         List[float] = []
    total_waiting:   List[float] = []   # per-step total cars waiting
    amb_delays:      List[int]   = []   # steps each ambulance waited
    steps_taken: int  = 0
    score:       float = 0.0
    success:     bool  = False
    crashed:     bool  = False
    current_amb_delay: int = 0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # ── reset ────────────────────────────────────────────────────────────
        resp = requests.post(
            f"{ENV_URL}/reset",
            json={"seed": seed, "task_id": task_id},
            timeout=30,
        )
        resp.raise_for_status()
        obs = obs_dict_to_array(resp.json())

        # ── step loop ────────────────────────────────────────────────────────
        for step in range(1, max_steps + 1):
            if client is not None:
                action_int, action_str = get_llm_action(client, obs)
            else:
                action_int = heuristic_action(obs)
                action_str = f"heuristic({action_int})"

            step_resp = requests.post(
                f"{ENV_URL}/step",
                json={"action": action_int},
                timeout=30,
            )
            step_resp.raise_for_status()
            result = step_resp.json()

            obs_dict    = result["observation"]
            obs         = obs_dict_to_array(obs_dict)
            reward      = float(result["reward"]["value"])
            done        = bool(result["done"])
            step_crashed = bool(result["info"].get("crashed", False))
            wait        = float(result["info"].get("total_waiting", 0.0))

            rewards.append(reward)
            total_waiting.append(wait)
            steps_taken = step

            if step_crashed:
                crashed = True

            # Track ambulance delay
            amb_present = bool(obs_dict["ambulance_ns"]) or bool(obs_dict["ambulance_ew"])
            if amb_present:
                current_amb_delay += 1
            else:
                if current_amb_delay > 0:
                    amb_delays.append(current_amb_delay)
                    current_amb_delay = 0

            error_str = "crashed" if step_crashed else None
            log_step(step=step, action=action_str, reward=reward, done=done, error=error_str)

            if done:
                break

        if current_amb_delay > 0:
            amb_delays.append(current_amb_delay)

        # pedestrian clearance: fraction of steps with 0 total waiting
        ped_clearance = sum(1 for w in total_waiting if w == 0) / max(steps_taken, 1)

        # ── grade via /grader endpoint ───────────────────────────────────────
        grade_resp = requests.post(
            f"{ENV_URL}/grader",
            json={
                "task_id": task_id,
                "episode_metrics": {
                    "steps_completed": steps_taken,
                    "crashed":         crashed,
                    "total_waiting":   total_waiting,
                    "amb_delays":      amb_delays if amb_delays else [0],
                    "ped_clearance":   ped_clearance,
                },
            },
            timeout=30,
        )
        if grade_resp.ok:
            score = float(grade_resp.json().get("score", 0.0))
        else:
            score = 0.0

        score   = float(np.clip(score, 0.0, 1.0))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} exception: {exc}", flush=True)
        if not rewards:
            rewards = [0.0]
        score   = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # Mandatory: use OpenAI Client configured via API_BASE_URL, MODEL_NAME, HF_TOKEN
    client: Optional[OpenAI] = None
    if API_KEY:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
            print(f"[DEBUG] LLM: {MODEL_NAME} via {API_BASE_URL}", flush=True)
        except Exception as exc:
            print(f"[DEBUG] OpenAI client init failed: {exc} — using heuristic", flush=True)
    else:
        print("[DEBUG] HF_TOKEN not set — using heuristic fallback policy", flush=True)

    for task in TASKS:
        run_task(client, task)


if __name__ == "__main__":
    main()
