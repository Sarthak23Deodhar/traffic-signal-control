"""
OpenEnv typed models for the Indian Traffic Control environment.
Defines Action, Observation, State, and Reward Pydantic models.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import uuid


# ─── Action Model ────────────────────────────────────────────────────────────

class TrafficAction(BaseModel):
    """
    Traffic signal control action.
    
    The agent selects one of 6 discrete actions to control the 4-way intersection.
    """
    action: int = Field(
        ...,
        ge=0,
        le=5,
        description=(
            "Discrete action: "
            "0=Keep current phase, "
            "1=Switch to N/S Straight Green, "
            "2=Switch to N/S Left-Turn Green, "
            "3=Switch to E/W Straight Green, "
            "4=Switch to E/W Left-Turn Green, "
            "5=Switch to All-Way Pedestrian Walk"
        )
    )

    class Config:
        json_schema_extra = {
            "example": {"action": 1},
        }


# ─── Observation Model ────────────────────────────────────────────────────────

class TrafficObservation(BaseModel):
    """
    A 12-dimensional observation snapshot of the intersection state.
    All vehicle counts are capped at max_cars (default 50).
    """
    cars_ns_straight: float = Field(..., ge=0.0, description="Cars queued to go straight North/South")
    cars_ns_left: float    = Field(..., ge=0.0, description="Cars queued to turn left  North/South")
    cars_ew_straight: float = Field(..., ge=0.0, description="Cars queued to go straight East/West")
    cars_ew_left: float    = Field(..., ge=0.0, description="Cars queued to turn left  East/West")
    ambulance_ns: float    = Field(..., ge=0.0, le=1.0, description="Emergency vehicle present on N/S lane (0 or 1)")
    ambulance_ew: float    = Field(..., ge=0.0, le=1.0, description="Emergency vehicle present on E/W lane (0 or 1)")
    pedestrians_waiting: float = Field(..., ge=0.0, description="Number of pedestrians waiting to cross")
    current_phase: float   = Field(..., ge=0.0, le=9.0, description="Current signal phase index (0–9)")
    time_in_phase: float   = Field(..., ge=0.0, description="Steps elapsed in the current phase")
    weather: float         = Field(..., ge=0.0, le=2.0, description="Weather: 0=Clear, 1=Rain, 2=Monsoon")
    pothole_ns: float      = Field(..., ge=0.0, le=1.0, description="Pothole/speedbreaker on N/S road (0 or 1)")
    pothole_ew: float      = Field(..., ge=0.0, le=1.0, description="Pothole/speedbreaker on E/W road (0 or 1)")

    class Config:
        json_schema_extra = {
            "example": {
                "cars_ns_straight": 5,
                "cars_ns_left": 2,
                "cars_ew_straight": 3,
                "cars_ew_left": 1,
                "ambulance_ns": 0,
                "ambulance_ew": 0,
                "pedestrians_waiting": 0,
                "current_phase": 0,
                "time_in_phase": 3,
                "weather": 0,
                "pothole_ns": 0,
                "pothole_ew": 1,
            }
        }


# ─── Reward Model ─────────────────────────────────────────────────────────────

class TrafficReward(BaseModel):
    """
    Step reward with a named breakdown for interpretability.
    Reward is dense (non-zero every step) to aid learning.
    """
    value: float = Field(..., description="Total scalar reward for this step")
    components: Dict[str, float] = Field(
        default_factory=dict,
        description="Named breakdown: wait_penalty, amb_penalty, amb_cleared, pedestrian_penalty, crash_penalty, phase_flicker_penalty"
    )


# ─── State Model ──────────────────────────────────────────────────────────────

class TrafficState(BaseModel):
    """
    Episode-level metadata (updated every step).
    """
    episode_id: str    = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique episode identifier")
    step_count: int    = Field(0, ge=0, description="Steps taken so far in the episode")
    total_reward: float = Field(0.0, description="Cumulative reward since episode start")
    done: bool         = Field(False, description="True when the episode has ended")
    crashed: bool      = Field(False, description="True if a collision has been detected")
    task_id: Optional[str] = Field(None, description="Active task ID, or None if free-play")
    max_steps: int     = Field(150, description="Maximum steps per episode")


# ─── Step Result (returned by step()) ────────────────────────────────────────

class StepResult(BaseModel):
    """
    Full result returned by POST /step.
    """
    observation: TrafficObservation
    reward: TrafficReward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
