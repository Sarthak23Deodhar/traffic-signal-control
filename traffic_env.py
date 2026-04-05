import gymnasium as gym
from gymnasium import spaces
import numpy as np

class IndianTrafficEnv(gym.Env):
    """
    ULTIMATE MASTERPIECE RL ENVIRONMENT: Indian Traffic Simulator.
    Features: Potholes, Speedbreakers, Erratic Speeds, Red-Light Runners, 
    Pedestrian Crossings, Weather, and Emergency Vehicles.
    """
    metadata = {'render.modes': ['console']}

    def __init__(self, steps_per_episode=150, max_cars=50):
        super(IndianTrafficEnv, self).__init__()
        self.steps_per_episode = steps_per_episode
        self.max_cars = max_cars
        self.current_step = 0
        
        # ACTIONS: 0=Keep Phase | 1=N/S Straight | 2=N/S Left | 3=E/W Straight | 4=E/W Left | 5=Pedestrian Walk
        self.action_space = spaces.Discrete(6)
        
        # STATE: 
        # 0: cars_ns_s, 1: cars_ns_l, 2: cars_ew_s, 3: cars_ew_l
        # 4: amb_ns, 5: amb_ew, 6: ped_waiting
        # 7: current_phase, 8: time_in_phase, 9: weather (Clear, Rain, Monsoon)
        # 10: pothole_ns (0/1), 11: pothole_ew (0/1)
        low = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([self.max_cars]*4 + [1, 1, 10, 10, 100, 2, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.phase_map = {
            0: "N/S Straight GREEN", 1: "N/S Straight YELLOW",
            2: "N/S Left Turn GREEN", 3: "N/S Left Turn YELLOW",
            4: "E/W Straight GREEN", 5: "E/W Straight YELLOW",
            6: "E/W Left Turn GREEN", 7: "E/W Left Turn YELLOW",
            8: "ALL-WAY PEDESTRIAN WALK", 9: "ALL RED (Transition)"
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        self.cars = np.random.randint(0, 5, size=4)
        self.amb_ns = 0; self.amb_ew = 0; self.ped_waiting = 0
        self.phase = 0; self.time_in_phase = 0
        self.weather = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1]) # 10% Intense Monsoon
        
        # Indian road features (Randomly spawned at start)
        self.pothole_ns = np.random.choice([0, 1], p=[0.5, 0.5]) # 50% chance of a pothole/speedbreaker N/S
        self.pothole_ew = np.random.choice([0, 1], p=[0.5, 0.5]) 
        self.target_phase = None
        self.crashed = False
        
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.cars[0], self.cars[1], self.cars[2], self.cars[3],
            self.amb_ns, self.amb_ew, self.ped_waiting,
            self.phase, self.time_in_phase, self.weather,
            self.pothole_ns, self.pothole_ew
        ], dtype=np.float32)

    def step(self, action):
        if self.crashed:
            # Episode is dead due to crash
            return self._get_obs(), 0.0, True, False, {'status': 'CRASHED'}
            
        reward = 0.0
        
        # 1. Spawning Dynamics
        for i in range(4): 
            if np.random.rand() < 0.2: self.cars[i] = min(self.cars[i] + np.random.randint(1, 4), self.max_cars)
        if np.random.rand() < 0.15: self.ped_waiting += 1
        if self.amb_ns == 0 and self.amb_ew == 0 and np.random.rand() < 0.02:
            if np.random.rand() < 0.5: self.amb_ns = 1
            else: self.amb_ew = 1

        # 2. Phase Transition Logic
        is_yellow = self.phase in [1, 3, 5, 7, 9]
        if is_yellow:
            if self.time_in_phase >= 2:
                self.phase = self.target_phase
                self.time_in_phase = 0
            else: self.time_in_phase += 1
        else:
            action_to_phase = {1: 0, 2: 2, 3: 4, 4: 6, 5: 8}
            if action != 0 and action in action_to_phase and action_to_phase[action] != self.phase:
                if self.time_in_phase < 2: reward -= 20.0 # Very high penalty for flickering
                else:
                    self.target_phase = action_to_phase[action]
                    self.phase = 9 if self.phase == 8 else self.phase + 1
                    self.time_in_phase = 0
            else: self.time_in_phase += 1

        # 3. Erratic Driver Speeds & Pothole Dynamics
        # Base departure rate depends on erratic driver behavior (randomized per step)
        base_rate = np.random.choice([1, 3, 5], p=[0.2, 0.6, 0.2]) # 20% slow driver, 60% normal, 20% speeding
        
        weather_penalty = {0: 0, 1: 1, 2: 2}[self.weather]
        dep_ns = max(0, base_rate - weather_penalty - (1 if self.pothole_ns else 0))
        dep_ew = max(0, base_rate - weather_penalty - (1 if self.pothole_ew else 0))

        # 4. Processing Departures & Rule Breaking
        # 5% chance a driver on RED jumps the signal!
        red_light_runner_ns = np.random.rand() < 0.05
        red_light_runner_ew = np.random.rand() < 0.05

        if self.phase == 0: # N/S Straight Green
            self.cars[0] = max(0, self.cars[0] - dep_ns)
            if self.amb_ns: self.amb_ns = 0; reward += 300.0
            
            # If EW driver jumps red while NS is Green -> CRASH!
            if red_light_runner_ew and (self.cars[2] > 0 or self.cars[3] > 0):
                self.crashed = True
                reward -= 1000.0
                
        elif self.phase == 2: # N/S Left
            self.cars[1] = max(0, self.cars[1] - dep_ns)
            if red_light_runner_ew and (self.cars[2] > 0 or self.cars[3] > 0):
                self.crashed = True
                reward -= 1000.0
                
        elif self.phase == 4: # E/W Straight Green
            self.cars[2] = max(0, self.cars[2] - dep_ew)
            if self.amb_ew: self.amb_ew = 0; reward += 300.0
            
            if red_light_runner_ns and (self.cars[0] > 0 or self.cars[1] > 0):
                self.crashed = True
                reward -= 1000.0
                
        elif self.phase == 6: # E/W Left
            self.cars[3] = max(0, self.cars[3] - dep_ew)
            if red_light_runner_ns and (self.cars[0] > 0 or self.cars[1] > 0):
                self.crashed = True
                reward -= 1000.0
                
        elif self.phase == 8: # Pedestrian Walk
            cleared = min(self.ped_waiting, 5)
            self.ped_waiting -= cleared
            reward += cleared * 15.0 

        # 5. General Wait Penalties
        reward -= np.sum(self.cars) * 1.5
        reward -= self.ped_waiting * 3.0
        if self.amb_ns and self.phase != 0: reward -= 100.0
        if self.amb_ew and self.phase != 4: reward -= 100.0

        self.current_step += 1
        terminated = self.crashed or bool(self.current_step >= self.steps_per_episode)
        
        info = {
            'total_waiting': np.sum(self.cars),
            'ped_waiting': self.ped_waiting,
            'phase_str': self.phase_map[self.phase],
            'crashed': self.crashed
        }
        
        return self._get_obs(), reward, terminated, False, info

    def render(self, mode='console'):
        if mode == 'console':
            if self.crashed: print("[CRASH OCCURRED!]"); return
            pt_str = f"Potholes [NS:{self.pothole_ns} EW:{self.pothole_ew}]"
            print(f"Step {self.current_step:03d} | {self.phase_map[self.phase]:<20} | {pt_str} | Wait: {np.sum(self.cars)}")
