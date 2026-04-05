"""
Baseline inference script for the Indian Traffic Control OpenEnv environment.

Uses the OpenAI API client to run a language model against the environment.
Reads API credentials from OPENAI_API_KEY environment variable.
Produces a reproducible baseline score on all 3 tasks.

Usage:
    export OPENAI_API_KEY=your_key_here
    python baseline.py

    # Or against a running Space:
    python baseline.py --base-url https://your-space.hf.space
"""
import os
import sys
import json
import argparse
import numpy as np

# ─── Attempt LLM baseline (requires OPENAI_API_KEY) ─────────────────────────

def build_obs_text(obs: np.ndarray) -> str:
    return (
        f"Intersection state:\n"
        f"  Cars waiting (N/S straight={obs[0]:.0f}, N/S left={obs[1]:.0f}, "
        f"E/W straight={obs[2]:.0f}, E/W left={obs[3]:.0f})\n"
        f"  Ambulance: N/S={'YES' if obs[4] else 'no'}, E/W={'YES' if obs[5] else 'no'}\n"
        f"  Pedestrians waiting: {obs[6]:.0f}\n"
        f"  Current signal phase: {int(obs[7])} (time in phase: {obs[8]:.0f})\n"
        f"  Weather: {['Clear','Rain','Monsoon'][int(obs[9])]}, "
        f"Potholes: N/S={'yes' if obs[10] else 'no'} E/W={'yes' if obs[11] else 'no'}\n"
        f"\nChoose the best traffic signal action (reply with a single integer):\n"
        f"  0 = Keep current phase\n"
        f"  1 = Switch to N/S Straight Green\n"
        f"  2 = Switch to N/S Left-Turn Green\n"
        f"  3 = Switch to E/W Straight Green\n"
        f"  4 = Switch to E/W Left-Turn Green\n"
        f"  5 = All-Way Pedestrian Walk\n"
        f"\nReturn ONLY the integer (0-5)."
    )


def llm_policy_factory(client, model: str):
    """Return a policy function that uses the LLM to select actions."""
    def policy(obs: np.ndarray) -> int:
        prompt = build_obs_text(obs)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": (
                        "You are an expert traffic signal controller AI. "
                        "Your goal is to minimize vehicle congestion, "
                        "prioritize emergency vehicles, and keep pedestrians safe."
                    )},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=5,
                temperature=0.0,
            )
            text = response.choices[0].message.content.strip()
            action = int(text[0])
            if 0 <= action <= 5:
                return action
        except Exception:
            pass
        return 0  # fallback: keep current phase
    return policy


def heuristic_policy(obs: np.ndarray) -> int:
    """
    Deterministic heuristic baseline (used when OPENAI_API_KEY is not set).
    Prioritises ambulances, then longest queue.
    """
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


# ─── Local episode runner ─────────────────────────────────────────────────────

def run_local(policy_fn, verbose: bool = True) -> dict:
    """Run all 3 tasks locally and return scores."""
    # Add project root to path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from server.tasks import run_task_episode, grade_episode

    results = {}
    for task_id in ["easy", "medium", "hard"]:
        metrics = run_task_episode(task_id, policy_fn)
        score = grade_episode(task_id, metrics)
        results[task_id] = {
            "score": round(score, 4),
            "steps": metrics["steps_completed"],
            "crashed": metrics["crashed"],
        }
        if verbose:
            print(f"  [{task_id:6s}] score={score:.4f} | steps={metrics['steps_completed']} | crashed={metrics['crashed']}")
    return results


# ─── HTTP-based runner (against a live HF Space) ─────────────────────────────

def run_against_server(base_url: str, policy_fn, verbose: bool = True) -> dict:
    """Run all 3 tasks against a running OpenEnv HTTP server."""
    try:
        import requests
    except ImportError:
        print("requests not installed — pip install requests")
        sys.exit(1)

    import numpy as np
    from server.tasks import grade_episode

    results = {}
    task_configs = {
        "easy":   {"seed": 42,  "max_steps": 50},
        "medium": {"seed": 123, "max_steps": 80},
        "hard":   {"seed": 999, "max_steps": 150},
    }

    for task_id, cfg in task_configs.items():
        resp = requests.post(f"{base_url}/reset", json={"seed": cfg["seed"], "task_id": task_id})
        resp.raise_for_status()
        obs_dict = resp.json()

        obs = np.array([
            obs_dict["cars_ns_straight"], obs_dict["cars_ns_left"],
            obs_dict["cars_ew_straight"], obs_dict["cars_ew_left"],
            obs_dict["ambulance_ns"], obs_dict["ambulance_ew"],
            obs_dict["pedestrians_waiting"], obs_dict["current_phase"],
            obs_dict["time_in_phase"], obs_dict["weather"],
            obs_dict["pothole_ns"], obs_dict["pothole_ew"],
        ], dtype=np.float32)

        total_waiting = []
        amb_delays = []
        current_amb_delay = 0
        crashed = False
        steps_done = 0

        for _ in range(cfg["max_steps"]):
            action = policy_fn(obs)
            resp = requests.post(f"{base_url}/step", json={"action": int(action)})
            resp.raise_for_status()
            result = resp.json()

            obs_dict = result["observation"]
            obs = np.array([
                obs_dict["cars_ns_straight"], obs_dict["cars_ns_left"],
                obs_dict["cars_ew_straight"], obs_dict["cars_ew_left"],
                obs_dict["ambulance_ns"], obs_dict["ambulance_ew"],
                obs_dict["pedestrians_waiting"], obs_dict["current_phase"],
                obs_dict["time_in_phase"], obs_dict["weather"],
                obs_dict["pothole_ns"], obs_dict["pothole_ew"],
            ], dtype=np.float32)

            total_waiting.append(result["info"].get("total_waiting", 0.0))
            steps_done += 1

            if obs_dict["ambulance_ns"] or obs_dict["ambulance_ew"]:
                current_amb_delay += 1
            else:
                if current_amb_delay > 0:
                    amb_delays.append(current_amb_delay)
                    current_amb_delay = 0

            if result["info"].get("crashed"):
                crashed = True
                break
            if result["done"]:
                break

        if current_amb_delay > 0:
            amb_delays.append(current_amb_delay)

        ped_clearance = sum(1 for w in total_waiting if w == 0) / max(steps_done, 1)
        metrics = {
            "task_id": task_id,
            "steps_completed": steps_done,
            "crashed": crashed,
            "total_waiting": total_waiting,
            "amb_delays": amb_delays or [0],
            "ped_clearance": ped_clearance,
        }
        score = grade_episode(task_id, metrics)
        results[task_id] = {"score": round(score, 4), "steps": steps_done, "crashed": crashed}

        if verbose:
            print(f"  [{task_id:6s}] score={score:.4f} | steps={steps_done} | crashed={crashed}")

    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OpenEnv Baseline Inference Script")
    parser.add_argument(
        "--base-url",
        default=None,
        help="URL of a running OpenEnv server (e.g. https://your-space.hf.space). If omitted, runs locally.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use for LLM baseline (default: gpt-4o-mini)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")

    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            policy = llm_policy_factory(client, args.model)
            print(f"Running LLM baseline with model={args.model}")
        except ImportError:
            print("openai package not found — falling back to heuristic baseline")
            policy = heuristic_policy
    else:
        print("OPENAI_API_KEY not set — running deterministic heuristic baseline")
        policy = heuristic_policy

    print("\n=== Indian Traffic Control — Baseline Scores ===")

    if args.base_url:
        print(f"Mode: HTTP against {args.base_url}\n")
        results = run_against_server(args.base_url, policy)
    else:
        print("Mode: local simulation\n")
        results = run_local(policy)

    print("\n─── Final Scores ───")
    for task_id, r in results.items():
        print(f"  {task_id:6s}: {r['score']:.4f}")

    print("\nBaseline scores (reproducible):")
    print(json.dumps({k: v["score"] for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
