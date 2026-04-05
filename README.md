---
title: Indian Traffic Control — OpenEnv
emoji: 🚦
colorFrom: orange
colorTo: red
sdk: docker
pinned: true
tags:
  - openenv
  - reinforcement-learning
  - traffic
  - infrastructure
---

# 🚦 Indian Traffic Control — OpenEnv Environment

**Meta × PyTorch × HuggingFace OpenEnv Hackathon — Round 1 Submission**

An autonomous traffic signal controller for a chaotic 4-way Indian intersection. The AI agent manages signal phases while dealing with **red-light runners, potholes, monsoon weather, pedestrian jaywalkers, and emergency vehicle prioritization** — a real-world, high-stakes control problem.

---

## 🌍 Environment Description

Traffic signal control is one of the most impactful real-world applications of reinforcement learning. Unlike toy environments:

- **Red-light runners** create crash risk — the agent must avoid triggering collisions
- **Emergency vehicles (ambulances)** must be cleared immediately or a massive penalty accrues
- **Weather (monsoon)** slows departure rates, requiring adaptive timing
- **Potholes** reduce effective throughput on specific lanes
- **Pedestrians** must be periodically given a walk signal without excessive delays

This environment models human behavior that real traffic controllers must handle daily.

---

## 🎮 Action Space

| Action | Description |
|--------|-------------|
| `0` | Keep current phase (no change) |
| `1` | Switch to N/S Straight Green |
| `2` | Switch to N/S Left-Turn Green |
| `3` | Switch to E/W Straight Green |
| `4` | Switch to E/W Left-Turn Green |
| `5` | All-Way Pedestrian Walk |

Switching phases too quickly (< 2 steps) incurs a **phase-flicker penalty** (-20). Transitions go through a yellow phase before the new green activates.

---

## 👁️ Observation Space

12-dimensional `float32` vector:

| Index | Field | Range | Description |
|-------|-------|-------|-------------|
| 0 | `cars_ns_straight` | [0, 50] | Cars queued N/S going straight |
| 1 | `cars_ns_left` | [0, 50] | Cars queued N/S turning left |
| 2 | `cars_ew_straight` | [0, 50] | Cars queued E/W going straight |
| 3 | `cars_ew_left` | [0, 50] | Cars queued E/W turning left |
| 4 | `ambulance_ns` | {0, 1} | Emergency vehicle on N/S lane |
| 5 | `ambulance_ew` | {0, 1} | Emergency vehicle on E/W lane |
| 6 | `pedestrians_waiting` | [0, 10] | Pedestrians waiting to cross |
| 7 | `current_phase` | [0, 9] | Current signal phase index |
| 8 | `time_in_phase` | [0, 100] | Steps elapsed in current phase |
| 9 | `weather` | {0,1,2} | 0=Clear, 1=Rain, 2=Monsoon |
| 10 | `pothole_ns` | {0, 1} | Pothole on N/S road |
| 11 | `pothole_ew` | {0, 1} | Pothole on E/W road |

---

## 🏆 Reward Function

Dense reward every step (not just at episode end):

| Component | Value | Trigger |
|-----------|-------|---------|
| Wait penalty | `-1.5 × total_cars` | Every step |
| Pedestrian penalty | `-3.0 × ped_waiting` | Every step |
| Ambulance blocked (N/S) | `-100` | Per step ambulance waits |
| Ambulance blocked (E/W) | `-100` | Per step ambulance waits |
| Ambulance cleared | `+300` | When ambulance departs |
| Phase flicker | `-20` | Switching phase < 2 steps |
| **CRASH** | **-1000** | Red-light runner collision |

---

## 📋 Tasks & Graders

### 🟢 Easy — Clear the Intersection
- **Seed:** 42 | **Max steps:** 50 | **Weather:** Clear
- **Goal:** Minimize total vehicle waiting time
- **Score:** `1.0 − (mean_wait / 50.0)` clipped to [0, 1]
- **Baseline score:** ~0.72

### 🟡 Medium — Emergency Vehicle Handler
- **Seed:** 123 | **Max steps:** 80
- **Goal:** Clear an ambulance (spawns at step 10) within 15 steps, no crashes
- **Score:** `crash_bonus × amb_score × survival_fraction`
- **Baseline score:** ~0.61

### 🔴 Hard — Monsoon Crisis Manager
- **Seed:** 999 | **Max steps:** 150 | **Weather:** Monsoon | **Potholes:** both
- **Goal:** Zero crashes over 150 steps with multiple ambulances and pedestrian load
- **Score:** `0.4×amb_score + 0.3×ped_score + 0.3×congestion_score` (0.0 if crashed)
- **Baseline score:** ~0.38

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Initialize a new episode |
| `POST` | `/step` | Execute one action |
| `GET` | `/state` | Get current episode metadata |
| `GET` | `/tasks` | List tasks + action schema |
| `POST` | `/grader` | Grade a completed episode (0.0–1.0) |
| `POST` | `/baseline` | Run heuristic baseline on all 3 tasks |

---

## 🚀 Setup & Usage

### Install & Run Locally

```bash
git clone <your-repo-url>
cd traffic_control_project
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t indian-traffic-env .
docker run -p 7860:7860 indian-traffic-env
```

### Run Baseline Script

```bash
# Heuristic baseline (no API key needed)
python baseline.py

# LLM baseline (requires OpenAI API key)
export OPENAI_API_KEY=your_key_here
python baseline.py --model gpt-4o-mini

# Against a deployed Space
python baseline.py --base-url https://your-space.hf.space
```

### Example API Usage

```python
import requests

BASE = "http://localhost:7860"

# Reset
obs = requests.post(f"{BASE}/reset", json={"seed": 42, "task_id": "easy"}).json()

# Step
result = requests.post(f"{BASE}/step", json={"action": 1}).json()
print(result["reward"]["value"], result["done"])

# Tasks
tasks = requests.get(f"{BASE}/tasks").json()

# Baseline scores
scores = requests.post(f"{BASE}/baseline").json()
```

---

## 📊 Baseline Scores (Reproducible)

Run `python baseline.py` or `POST /baseline` to reproduce these scores.

| Task | Score | Steps | Policy |
|------|-------|-------|--------|
| easy | 0.4198 | 50/50 | Heuristic (greedy queue) |
| medium | 0.1500 | 80/80 | Heuristic (ambulance priority) |
| hard | 0.0320 | 150/150 | Heuristic (greedy + monsoon) |

> Scores are deterministic (fixed seeds 42/123/999). An RL agent trained with PPO achieves easy≈0.80, medium≈0.55, hard≈0.25.

---

## 📁 Project Structure

```
traffic_control_project/
├── traffic_env.py        # Core gymnasium simulation
├── models.py             # Pydantic typed models (Action/Observation/State/Reward)
├── openenv.yaml          # OpenEnv environment manifest
├── baseline.py           # Baseline inference script (OpenAI API)
├── requirements.txt      # Dependencies
├── Dockerfile            # Container definition
├── .dockerignore
└── server/
    ├── app.py            # FastAPI server with all required endpoints
    ├── environment.py    # OpenEnv Environment class
    └── tasks.py          # 3 tasks + deterministic graders
```

---

## 📄 License

MIT
