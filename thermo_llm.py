"""
Thermo-LLM: NS-PIRL Framework
================================
Complete implementation of the Neuro-Symbolic Physics-Informed
Reinforcement Learning framework for thermal-aware LLM scheduling.

This script implements:
  1. RC Thermal Circuit Digital Twin (Physics-Informed)
  2. Phase-Aware Observer (simulated)
  3. Safety-Constrained D-DQN Scheduler
  4. Safety Shield with formal veto mechanism
  5. Full simulation over 5400-second inference session
  6. All baselines: ALL_FAST, ALL_SAFE, PHASE_AWARE, RL_ONLY
  7. Ablation study configurations
  8. Result generation matching the paper figures/tables

Authors: Sangeeth S, Jaganath R, Niranjan S, Kowshik P L
"""

import numpy as np
import random
import json
import os
from collections import deque
from datetime import datetime

# ============================================================
# CONFIGURATION — All constants from the paper
# ============================================================

class Config:
    """System parameters matching Table 4 of the paper."""
    
    # Thermal parameters (calibrated via offline least-squares fitting)
    R_TH = 4.2        # Thermal resistance (°C/W) — BCM2712 calibrated
    C_TH = 38.0       # Thermal capacitance (J/°C) — BCM2712 calibrated
    T_AMB = 28.0       # Ambient temperature (°C)
    T_LIMIT = 43.0     # Proactive safety threshold (°C)
    T_HW = 52.0        # Hardware throttle threshold (°C)
    
    # Frequency action space (MHz) — BCM2712 cpufreq steps
    FREQ_ACTIONS = [600, 900, 1200, 1500, 1800, 2400]
    N_ACTIONS = len(FREQ_ACTIONS)
    
    # Power model parameters
    # P = kappa * V^2 * f + P_leak
    # Simplified: P(f, phase) = base_power(phase) * (f / f_max)^alpha
    P_PREFILL_MAX = 7.5   # Watts at 2400 MHz during prefill
    P_DECODE_MAX = 5.8    # Watts at 2400 MHz during decode
    P_LEAK = 0.8          # Leakage power (Watts)
    POWER_ALPHA = 1.8     # Frequency-power scaling exponent
    
    # LLM workload parameters (Gemma 2, 2.0B, INT4)
    PREFILL_DURATION_RANGE = (2.0, 6.0)  # seconds
    DECODE_DURATION_RANGE = (15.0, 90.0)  # seconds
    
    # Throughput model: tokens/sec as function of frequency
    TPS_MAX = 9.8          # At 2400 MHz (Gemma 2)
    TPS_MIN = 3.2          # At 600 MHz
    
    # Accuracy model
    A_GPU = 52.4           # MMLU % with GPU (FP32)
    A_NPU = 50.3           # MMLU % with NPU (INT4) — epsilon_acc ≈ 2.1%
    EPSILON_ACC = A_GPU - A_NPU  # 2.1%
    
    # RL hyperparameters
    GAMMA = 0.99           # Discount factor
    LR = 1e-4              # Learning rate
    BATCH_SIZE = 64
    REPLAY_BUFFER_SIZE = 100000
    TARGET_UPDATE_FREQ = 50
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.9995
    ALPHA_REWARD = 0.7     # Throughput vs accuracy weight
    PENALTY_THERMAL = 100.0
    
    # Simulation
    DT = 0.5               # Discretization timestep (seconds)
    LOOKAHEAD_STEPS = 20   # 20 steps * 0.5s = 10-second look-ahead
    SESSION_LENGTH = 5400  # 90 minutes
    SCHEDULING_INTERVAL = 1.0  # Decision every 1 second

    # Seed for reproducibility
    SEED = 42


# ============================================================
# MODULE 1: RC THERMAL CIRCUIT — Physics-Informed Digital Twin
# ============================================================

class ThermalDigitalTwin:
    """
    Physics-Informed Digital Twin using first-order RC thermal circuit.
    
    Implements Eq. (1) from the paper:
      dT/dt = (1/C_th) * [P_in - (T - T_amb) / R_th]
    
    Calibrated via offline least-squares fitting on real telemetry.
    Online Kalman filter for drift compensation.
    """
    
    def __init__(self, R_th=Config.R_TH, C_th=Config.C_TH, T_amb=Config.T_AMB):
        self.R_th = R_th
        self.C_th = C_th
        self.T_amb = T_amb
        
        # Kalman filter state for online drift compensation
        self.kalman_gain = 0.01
        self.R_th_estimate = R_th
        self.C_th_estimate = C_th
    
    def step(self, T_current, P_in, dt=Config.DT):
        """
        Single-step forward Euler integration of the heat equation.
        
        Eq. (7): T_{k+1} = T_k + (dt/C_th) * [P_in - (T_k - T_amb)/R_th]
        """
        dT = (dt / self.C_th_estimate) * (
            P_in - (T_current - self.T_amb) / self.R_th_estimate
        )
        return T_current + dT
    
    def forecast(self, T_current, P_in, n_steps=Config.LOOKAHEAD_STEPS, 
                 dt=Config.DT):
        """
        Generate n-step thermal forecast vector.
        
        Iterates Eq. (7) for k=1..n_steps to produce the
        10-second thermal trajectory T_hat = [T_1, ..., T_20].
        
        Returns:
            trajectory: list of predicted temperatures
        """
        trajectory = []
        T = T_current
        for _ in range(n_steps):
            T = self.step(T, P_in, dt)
            trajectory.append(T)
        return trajectory
    
    def kalman_update(self, T_predicted, T_measured):
        """
        Lightweight Kalman filter for online drift compensation.
        Compensates for ambient temperature changes and aging.
        """
        error = T_measured - T_predicted
        # Small correction to R_th estimate
        self.R_th_estimate += self.kalman_gain * error * 0.01
        # Bound to physical limits
        self.R_th_estimate = np.clip(self.R_th_estimate, 2.0, 8.0)
    
    def physics_loss(self, T_pred_series, P_series, T_actual_series, dt=Config.DT):
        """
        Physics-informed loss for PINN training.
        
        Eq. (5): L_physics = ||dT_hat/dt - (1/C_th)(P_in - (T_hat-T_amb)/R_th)||^2
        
        This penalizes predictions that violate the heat diffusion equation.
        """
        loss = 0.0
        for i in range(len(T_pred_series) - 1):
            dT_dt_pred = (T_pred_series[i+1] - T_pred_series[i]) / dt
            dT_dt_physics = (1.0 / self.C_th) * (
                P_series[i] - (T_pred_series[i] - self.T_amb) / self.R_th
            )
            loss += (dT_dt_pred - dT_dt_physics) ** 2
        return loss / max(len(T_pred_series) - 1, 1)


# ============================================================
# MODULE 2: PHASE-AWARE OBSERVER
# ============================================================

class PhaseAwareObserver:
    """
    Simulates the phase-aware observer hooked into llama.cpp.
    
    In the real implementation, this is a C++ module that:
    - Detects KV-cache initialization → φ=1 (Prefill start)
    - Detects first token emission → φ=0 (Decode start)
    - Sends binary signal via Unix socket to the scheduling daemon
    
    For simulation, we model the bimodal LLM inference pattern.
    """
    
    def __init__(self, seed=Config.SEED):
        self.rng = np.random.RandomState(seed)
        self.phase = 0          # 0=Decode, 1=Prefill
        self.phase_timer = 0.0  # Time remaining in current phase
        self.query_count = 0    # Number of inference queries
        self._generate_next_phase()
    
    def _generate_next_phase(self):
        """Generate timing for next Prefill→Decode cycle."""
        self.phase = 1  # Start with Prefill
        self.prefill_duration = self.rng.uniform(*Config.PREFILL_DURATION_RANGE)
        self.decode_duration = self.rng.uniform(*Config.DECODE_DURATION_RANGE)
        self.phase_timer = self.prefill_duration
        self.query_count += 1
    
    def step(self, dt=Config.DT):
        """
        Advance the phase observer by dt seconds.
        
        Returns:
            phase: current phase (0=Decode, 1=Prefill)
            boundary: True if a φ: 1→0 transition just occurred
        """
        self.phase_timer -= dt
        boundary = False
        
        if self.phase_timer <= 0:
            if self.phase == 1:
                # Prefill→Decode transition (the critical boundary)
                self.phase = 0
                self.phase_timer = self.decode_duration
                boundary = True
            else:
                # Decode finished → next query starts with Prefill
                self._generate_next_phase()
        
        return self.phase, boundary
    
    def get_phase_signal(self):
        """Returns binary phase signal φ(t) ∈ {0, 1}."""
        return self.phase


# ============================================================
# MODULE 3: POWER AND THROUGHPUT MODELS
# ============================================================

class PowerModel:
    """
    Models power consumption and throughput as functions of
    frequency, phase, and hardware mode.
    
    Power: P = kappa_phase * V^2 * f + P_leak  (Eq. 9)
    Throughput: τ = TPS_max * (f / f_max)^0.7   (sub-linear scaling)
    """
    
    @staticmethod
    def compute_power(freq_mhz, phase, hardware_mode='GPU'):
        """
        Compute instantaneous power draw.
        
        Args:
            freq_mhz: CPU frequency in MHz
            phase: 0=Decode, 1=Prefill
            hardware_mode: 'GPU' or 'NPU'
        """
        f_ratio = freq_mhz / Config.FREQ_ACTIONS[-1]  # Normalize to max freq
        
        if hardware_mode == 'NPU':
            # NPU: much lower power, fixed efficiency
            base_power = 4.1  # Watts (NPU typical)
            return base_power * (f_ratio ** 0.5) + Config.P_LEAK * 0.5
        
        if phase == 1:  # Prefill
            base_power = Config.P_PREFILL_MAX
        else:  # Decode
            base_power = Config.P_DECODE_MAX
        
        return base_power * (f_ratio ** Config.POWER_ALPHA) + Config.P_LEAK
    
    @staticmethod
    def compute_throughput(freq_mhz, phase, hardware_mode='GPU'):
        """
        Compute instantaneous token throughput.
        
        During Prefill, throughput = 0 (processing prompt).
        During Decode, throughput scales sub-linearly with frequency.
        """
        if phase == 1:  # Prefill — no tokens emitted
            return 0.0
        
        f_ratio = freq_mhz / Config.FREQ_ACTIONS[-1]
        
        if hardware_mode == 'NPU':
            # NPU: slightly lower throughput due to INT4
            return Config.TPS_MAX * 0.92 * (f_ratio ** 0.7)
        
        return Config.TPS_MAX * (f_ratio ** 0.7)
    
    @staticmethod
    def compute_accuracy(hardware_mode='GPU'):
        """Accuracy under hardware mode H. Eq. (2)."""
        if hardware_mode == 'GPU':
            return Config.A_GPU
        else:
            return Config.A_NPU


# ============================================================
# MODULE 4: SAFETY SHIELD (Symbolic Veto Mechanism)
# ============================================================

class SafetyShield:
    """
    Symbolic Safety Shield implementing the formal veto mechanism.
    
    Section 4.3.3 of the paper:
    1. D-DQN proposes f_cand
    2. Digital Twin forecasts T_hat for 10 seconds
    3. If max(T_hat) < T_limit → APPROVE
    4. If max(T_hat) >= T_limit → VETO:
       Find highest f_safe where max(T_hat(f_safe)) < T_limit
    5. Emergency: if no safe f exists → f_min
    """
    
    def __init__(self, twin: ThermalDigitalTwin, ablation_cfg=None):
        self.twin = twin
        self.ablation_cfg = ablation_cfg or {
            'shield': True, 'twin': True, 'phase': True, 'boundary': True
        }
        self.veto_count = 0
        self.approval_count = 0
    
    def evaluate(self, T_current, phase, f_cand_idx, hardware_mode='GPU'):
        """
        Evaluate a candidate frequency against the thermal forecast.
        
        Returns:
            safe_action_idx: approved frequency index
            was_vetoed: True if the original action was vetoed
            forecast: the thermal forecast for the approved action
        """
        f_cand = Config.FREQ_ACTIONS[f_cand_idx]
        P_cand = PowerModel.compute_power(f_cand, phase, hardware_mode)
        
        if self.ablation_cfg.get('twin', True):
            forecast_cand = self.twin.forecast(T_current, P_cand)
        else:
            forecast_cand = [T_current] * Config.LOOKAHEAD_STEPS
        
        # Approval check: Eq. (10)
        if max(forecast_cand) < Config.T_LIMIT:
            self.approval_count += 1
            return f_cand_idx, False, forecast_cand
        
        # VETO — Search for highest safe frequency (descending order)
        self.veto_count += 1
        
        for idx in range(len(Config.FREQ_ACTIONS) - 1, -1, -1):
            f_test = Config.FREQ_ACTIONS[idx]
            P_test = PowerModel.compute_power(f_test, phase, hardware_mode)
            if self.ablation_cfg.get('twin', True):
                forecast_test = self.twin.forecast(T_current, P_test)
            else:
                forecast_test = [T_current] * Config.LOOKAHEAD_STEPS
            
            if max(forecast_test) < Config.T_LIMIT:
                return idx, True, forecast_test
        
        # Emergency fallback: f_min
        P_min = PowerModel.compute_power(Config.FREQ_ACTIONS[0], phase, hardware_mode)
        if self.ablation_cfg.get('twin', True):
            forecast_min = self.twin.forecast(T_current, P_min)
        else:
            forecast_min = [T_current] * Config.LOOKAHEAD_STEPS
        return 0, True, forecast_min


# ============================================================
# MODULE 5: D-DQN AGENT
# ============================================================

class ReplayBuffer:
    """Experience replay buffer for D-DQN training."""
    
    def __init__(self, capacity=Config.REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size=Config.BATCH_SIZE):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class SimpleQNetwork:
    """
    Lightweight Q-Network implemented with NumPy.
    
    Architecture (from paper Section 4.3.2):
    Input(14) → FC(128, ReLU) → FC(64, ReLU) → FC(6, linear)
    
    Using NumPy instead of PyTorch for portability.
    In deployment, this is converted to TFLite INT8.
    """
    
    def __init__(self, input_dim=14, hidden1=128, hidden2=64, 
                 output_dim=Config.N_ACTIONS, seed=Config.SEED):
        rng = np.random.RandomState(seed)
        # Xavier initialization
        self.W1 = rng.randn(input_dim, hidden1) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden1)
        self.W2 = rng.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros(hidden2)
        self.W3 = rng.randn(hidden2, output_dim) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros(output_dim)
    
    def forward(self, x):
        """Forward pass with ReLU activations."""
        h1 = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)  # ReLU
        q_values = h2 @ self.W3 + self.b3            # Linear output
        return q_values
    
    def predict(self, state):
        """Get Q-values for a single state."""
        if state.ndim == 1:
            state = state.reshape(1, -1)
        return self.forward(state)[0]
    
    def copy_from(self, other):
        """Polyak averaging for target network update."""
        tau = 0.01
        self.W1 = tau * other.W1 + (1 - tau) * self.W1
        self.b1 = tau * other.b1 + (1 - tau) * self.b1
        self.W2 = tau * other.W2 + (1 - tau) * self.W2
        self.b2 = tau * other.b2 + (1 - tau) * self.b2
        self.W3 = tau * other.W3 + (1 - tau) * self.W3
        self.b3 = tau * other.b3 + (1 - tau) * self.b3
    
    def update(self, states, targets, lr=Config.LR):
        """Simple gradient descent update."""
        # Forward pass
        h1 = np.maximum(0, states @ self.W1 + self.b1)
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)
        q_pred = h2 @ self.W3 + self.b3
        
        # Backward pass (simplified)
        error = q_pred - targets
        grad_W3 = h2.T @ error / len(states)
        grad_b3 = error.mean(axis=0)
        
        # Update
        self.W3 -= lr * grad_W3
        self.b3 -= lr * grad_b3


class DDQN_Agent:
    """
    Double Deep Q-Network agent with Safety Shield.
    
    Implements the SC-D-DQN described in Section 4.3 of the paper.
    Action selection is decoupled from evaluation to prevent
    overestimation bias (critical for safety-sensitive scheduling).
    """
    
    def __init__(self, twin: ThermalDigitalTwin, seed=Config.SEED, ablation_cfg=None):
        self.twin = twin
        self.ablation_cfg = ablation_cfg or {
            'shield': True, 'twin': True, 'phase': True, 'boundary': True
        }
        self.shield = SafetyShield(twin) if self.ablation_cfg.get('shield', True) else None
        self.q_network = SimpleQNetwork(seed=seed)
        self.target_network = SimpleQNetwork(seed=seed + 1)
        self.target_network.copy_from(self.q_network)
        self.replay_buffer = ReplayBuffer()
        self.epsilon = Config.EPSILON_START
        self.step_count = 0
        self.hardware_mode = 'GPU'
        self.necessary_failure_count = 0
        self.rng = np.random.RandomState(seed)
    
    def build_state(self, T_curr, freq_idx, P_inst, phase, forecast):
        """
        Construct state vector s_t = {T_curr, f_clk, P_inst, φ, T_hat[t:t+D]}
        
        Eq. (7) from the paper. Total dimension: 4 + 10 = 14
        (using every other forecast step for compression to 10 values)
        """
        state = np.zeros(14)
        state[0] = (T_curr - Config.T_AMB) / (Config.T_HW - Config.T_AMB)  # Normalized
        state[1] = freq_idx / (Config.N_ACTIONS - 1)  # Normalized
        state[2] = P_inst / Config.P_PREFILL_MAX  # Normalized
        state[3] = float(phase)
        # 10-step forecast (every other step from 20-step forecast)
        for i in range(10):
            idx = min(i * 2, len(forecast) - 1)
            state[4 + i] = (forecast[idx] - Config.T_AMB) / (Config.T_HW - Config.T_AMB)
        return state
    
    def select_action(self, state, T_curr, phase):
        """
        Epsilon-greedy action selection with Safety Shield.
        
        1. D-DQN proposes f_cand
        2. Safety Shield evaluates thermal feasibility
        3. If vetoed, highest safe f is selected instead
        """
        # Epsilon-greedy exploration
        if self.rng.random() < self.epsilon:
            f_cand_idx = self.rng.randint(0, Config.N_ACTIONS)
        else:
            q_values = self.q_network.predict(state)
            f_cand_idx = np.argmax(q_values)
        
        # Safety Shield evaluation (Section 4.3.3)
        if self.shield is not None:
            safe_idx, was_vetoed, forecast = self.shield.evaluate(
                T_curr, phase, f_cand_idx, self.hardware_mode
            )
        else:
            safe_idx = f_cand_idx
            was_vetoed = False
            P_cand = PowerModel.compute_power(Config.FREQ_ACTIONS[f_cand_idx], phase, self.hardware_mode)
            forecast = self.twin.forecast(T_curr, P_cand)
        
        return safe_idx, was_vetoed, forecast
    
    def check_necessary_failure(self, T_curr, phase, boundary):
        """
        Check the 3 conditions for "Necessary Failure" (Section 4.4).
        
        Triggers when ALL THREE are met:
        1. φ(t) = 0 (at Prefill→Decode boundary)
        2. max(T_hat_GPU) >= T_limit (GPU will violate)
        3. max(T_hat_NPU) < T_limit (NPU is thermally safe)
        
        Returns True if hardware mode should switch to NPU.
        """
        if not boundary:
            return False
        
        if self.hardware_mode == 'NPU':
            # Already on NPU — check if we can return to GPU
            P_gpu = PowerModel.compute_power(Config.FREQ_ACTIONS[-1], 0, 'GPU')
            if self.ablation_cfg.get('twin', True):
                forecast_gpu = self.twin.forecast(T_curr, P_gpu)
            else:
                forecast_gpu = [T_curr] * Config.LOOKAHEAD_STEPS
            if max(forecast_gpu) < Config.T_LIMIT - 2.0:  # Hysteresis
                self.hardware_mode = 'GPU'
            return False
        
        # Condition 1: At boundary (already checked above)
        # Condition 2: GPU will violate
        P_gpu = PowerModel.compute_power(Config.FREQ_ACTIONS[3], 0, 'GPU')
        if self.ablation_cfg.get('twin', True):
            forecast_gpu = self.twin.forecast(T_curr, P_gpu)
        else:
            forecast_gpu = [T_curr] * Config.LOOKAHEAD_STEPS
        
        if max(forecast_gpu) < Config.T_LIMIT:
            return False  # GPU is safe — no need for Necessary Failure
        
        # Condition 3: NPU is thermally safe
        P_npu = PowerModel.compute_power(Config.FREQ_ACTIONS[3], 0, 'NPU')
        if self.ablation_cfg.get('twin', True):
            forecast_npu = self.twin.forecast(T_curr, P_npu)
        else:
            forecast_npu = [T_curr] * Config.LOOKAHEAD_STEPS
        
        if max(forecast_npu) < Config.T_LIMIT:
            self.hardware_mode = 'NPU'
            self.necessary_failure_count += 1
            return True
        
        return False
    
    def compute_reward(self, throughput, accuracy, T_max_forecast):
        """
        Reward function (Eq. 8):
        r = α·τ + (1-α)·A  if max(T_hat) < T_limit
        r = -P_thermal      if max(T_hat) >= T_limit
        """
        if T_max_forecast >= Config.T_LIMIT:
            return -Config.PENALTY_THERMAL
        
        # Normalize throughput and accuracy
        tau_norm = throughput / Config.TPS_MAX
        acc_norm = accuracy / 100.0
        
        return Config.ALPHA_REWARD * tau_norm + (1 - Config.ALPHA_REWARD) * acc_norm
    
    def train_step(self):
        """Single training step using experience replay."""
        if len(self.replay_buffer) < Config.BATCH_SIZE:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        
        # D-DQN: use online network to SELECT action, target to EVALUATE
        q_next_online = np.array([self.q_network.predict(s) for s in next_states])
        best_actions = np.argmax(q_next_online, axis=1)
        
        q_next_target = np.array([self.target_network.predict(s) for s in next_states])
        q_target_values = q_next_target[np.arange(len(best_actions)), best_actions]
        
        targets = np.array([self.q_network.predict(s) for s in states])
        for i in range(len(actions)):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + Config.GAMMA * q_target_values[i]
        
        self.q_network.update(states, targets)
        
        # Update target network
        self.step_count += 1
        if self.step_count % Config.TARGET_UPDATE_FREQ == 0:
            self.target_network.copy_from(self.q_network)
        
        # Decay epsilon
        self.epsilon = max(Config.EPSILON_END, 
                         self.epsilon * Config.EPSILON_DECAY)


# ============================================================
# SIMULATION ENGINE
# ============================================================

class ThermoLLMSimulator:
    """
    Complete simulation of the Thermo-LLM framework.
    Runs the closed-loop control (Algorithm 1) and logs all metrics.
    """
    
    def __init__(self, method='NS_PIRL', seed=Config.SEED, ablation_cfg=None):
        self.method = method
        self.seed = seed
        self.ablation_cfg = ablation_cfg or {
            'shield': True, 'twin': True, 'phase': True, 'boundary': True
        }
        np.random.seed(seed)
        random.seed(seed)
        
        # Initialize components
        self.twin = ThermalDigitalTwin()
        self.observer = PhaseAwareObserver(seed=seed)
        
        if method == 'NS_PIRL':
            self.agent = DDQN_Agent(self.twin, seed=seed, ablation_cfg=self.ablation_cfg)
        else:
            self.agent = None
        
        # System state
        self.T_current = Config.T_AMB + 4.0  # Start at ~32°C
        self.freq_idx = 5  # Start at max frequency
        self.hardware_mode = 'GPU'
        
        # Logging
        self.log = {
            'time': [],
            'temperature': [],
            'frequency': [],
            'power': [],
            'throughput': [],
            'phase': [],
            'hardware_mode': [],
            'violations': [],
            'utility': [],
        }
        self.total_violations = 0
        self.total_tokens = 0
        self.total_energy = 0.0
    
    def _select_action_baseline(self, phase):
        """Action selection for baseline methods."""
        if self.method == 'ALL_FAST':
            return 5  # Always max frequency
        
        elif self.method == 'ALL_SAFE':
            return 0  # Always min frequency
        
        elif self.method == 'PHASE_AWARE':
            # Simple rule: high freq during decode, lower during prefill
            if phase == 1:  # Prefill
                return 3  # 1500 MHz
            else:  # Decode
                return 4  # 1800 MHz
        
        elif self.method == 'RL_ONLY':
            # Reactive RL without physics or safety shield
            # Simple policy: if hot, reduce; if cool, increase
            if self.T_current > Config.T_LIMIT:
                return max(0, self.freq_idx - 2)
            elif self.T_current > Config.T_LIMIT - 3:
                return max(0, self.freq_idx - 1)
            elif self.T_current < Config.T_LIMIT - 5:
                return min(5, self.freq_idx + 1)
            return self.freq_idx
        
        return 5
    
    def _apply_reactive_throttling(self):
        """
        Simulate Linux thermal governor emergency throttling.
        Triggered when T >= T_HW (52°C).
        """
        if self.T_current >= Config.T_HW:
            self.freq_idx = 0  # Emergency: minimum frequency
            return True
        elif self.T_current >= Config.T_LIMIT and self.method != 'NS_PIRL':
            # Governor starts reducing (for baseline methods)
            self.freq_idx = max(0, self.freq_idx - 1)
            return True
        return False
    
    def run(self, duration=Config.SESSION_LENGTH, training=True):
        """
        Run the complete simulation for the specified duration.
        
        This implements Algorithm 1 from the paper.
        """
        t = 0.0
        decision_timer = 0.0
        
        while t < duration:
            # Step 1: Get phase signal (Section 4.1)
            phase, boundary = self.observer.step(Config.DT)
            if not self.ablation_cfg.get('phase', True):
                phase = 0
                boundary = False
            
            # Step 2: Compute current power and throughput
            freq = Config.FREQ_ACTIONS[self.freq_idx]
            P_inst = PowerModel.compute_power(freq, phase, self.hardware_mode)
            tau = PowerModel.compute_throughput(freq, phase, self.hardware_mode)
            accuracy = PowerModel.compute_accuracy(self.hardware_mode)
            
            # Step 3: Update thermal state using physics model
            T_new = self.twin.step(self.T_current, P_inst, Config.DT)
            
            # Add small stochastic noise (sensor noise, convection)
            T_new += np.random.normal(0, 0.02)
            
            # Step 4: Check for hardware-level throttling (reactive baselines)
            violated = self.T_current >= Config.T_LIMIT
            if violated:
                self.total_violations += 1
            
            self._apply_reactive_throttling()
            
            # Step 5: Scheduling decision (every 1 second)
            decision_timer += Config.DT
            if decision_timer >= Config.SCHEDULING_INTERVAL:
                decision_timer = 0.0
                
                if self.method == 'NS_PIRL':
                    # NS-PIRL: Full predictive pipeline
                    
                    # Check Necessary Failure at boundaries (Section 4.4)
                    effective_boundary = boundary if self.ablation_cfg.get('boundary', True) else True
                    self.agent.check_necessary_failure(
                        self.T_current, phase, effective_boundary
                    )
                    self.hardware_mode = self.agent.hardware_mode
                    
                    # Get thermal forecast
                    P_forecast = PowerModel.compute_power(
                        freq, phase, self.hardware_mode
                    )
                    if self.ablation_cfg.get('twin', True):
                        forecast = self.twin.forecast(self.T_current, P_forecast)
                    else:
                        forecast = [self.T_current] * Config.LOOKAHEAD_STEPS
                    
                    # Build state vector
                    state = self.agent.build_state(
                        self.T_current, self.freq_idx, P_inst, phase, forecast
                    )
                    
                    # Select action with Safety Shield
                    new_freq_idx, vetoed, approved_forecast = self.agent.select_action(
                        state, self.T_current, phase
                    )
                    
                    # Compute reward
                    T_max_forecast = max(approved_forecast)
                    reward = self.agent.compute_reward(tau, accuracy, T_max_forecast)
                    
                    # Store transition and train
                    if training and hasattr(self, '_prev_state'):
                        self.agent.replay_buffer.push(
                            self._prev_state, self._prev_action, 
                            self._prev_reward, state, False
                        )
                        self.agent.train_step()
                    
                    self._prev_state = state
                    self._prev_action = new_freq_idx
                    self._prev_reward = reward
                    
                    self.freq_idx = new_freq_idx
                    
                else:
                    # Baseline methods
                    self.freq_idx = self._select_action_baseline(phase)
            
            # Step 6: Update system state
            self.T_current = T_new
            
            # Step 7: Accumulate metrics
            self.total_tokens += tau * Config.DT
            self.total_energy += P_inst * Config.DT
            
            # Compute utility
            utility = Config.ALPHA_REWARD * (tau / Config.TPS_MAX) + \
                     (1 - Config.ALPHA_REWARD) * (accuracy / 100.0)
            if violated and self.method != 'NS_PIRL':
                utility -= 5.0  # Penalty for violation
            
            # Log
            self.log['time'].append(t)
            self.log['temperature'].append(self.T_current)
            self.log['frequency'].append(Config.FREQ_ACTIONS[self.freq_idx])
            self.log['power'].append(P_inst)
            self.log['throughput'].append(tau)
            self.log['phase'].append(phase)
            self.log['hardware_mode'].append(self.hardware_mode)
            self.log['violations'].append(1 if violated else 0)
            self.log['utility'].append(utility)
            
            t += Config.DT
        
        return self._compute_summary()
    
    def _compute_summary(self):
        """Compute summary metrics matching the paper's tables."""
        temps = np.array(self.log['temperature'])
        tps_values = np.array(self.log['throughput'])
        
        # Filter only decode phase for STR calculation
        decode_tps = [tps for tps, ph in 
                      zip(self.log['throughput'], self.log['phase']) if ph == 0]
        
        summary = {
            'method': self.method,
            'violations': self.total_violations,
            'max_temperature': float(np.max(temps)),
            'mean_temperature': float(np.mean(temps)),
            'STR': float(np.mean(decode_tps)) if decode_tps else 0.0,
            'energy_per_token': float(self.total_energy / max(self.total_tokens, 1)),
            'total_tokens': float(self.total_tokens),
            'total_energy_J': float(self.total_energy),
        }
        
        if self.agent and getattr(self.agent, 'shield', None) is not None:
            summary['shield_vetoes'] = self.agent.shield.veto_count
            summary['shield_approvals'] = self.agent.shield.approval_count
        if self.agent and hasattr(self.agent, 'necessary_failure_count'):
            summary['necessary_failures'] = self.agent.necessary_failure_count
        
        return summary


# ============================================================
# MAIN: RUN ALL EXPERIMENTS
# ============================================================

def run_all_experiments():
    """
    Execute all experiments described in Section 8 of the paper.
    Runs NS-PIRL and all 4 baselines over a 5400-second session.
    """
    print("=" * 70)
    print("  THERMO-LLM: NS-PIRL FRAMEWORK — EXPERIMENTAL EVALUATION")
    print("=" * 70)
    print(f"  Session length: {Config.SESSION_LENGTH}s (90 minutes)")
    print(f"  Safety threshold T_limit: {Config.T_LIMIT}°C")
    print(f"  Hardware throttle T_hw: {Config.T_HW}°C")
    print(f"  Look-ahead horizon: {Config.LOOKAHEAD_STEPS * Config.DT}s")
    print(f"  Seed: {Config.SEED}")
    print("=" * 70)
    
    methods = ['NS_PIRL', 'ALL_FAST', 'ALL_SAFE', 'PHASE_AWARE', 'RL_ONLY']
    results = {}
    trained_agent = None
    
    for method in methods:
        print(f"\n{'-' * 50}")
        print(f"  Running: {method}")
        print(f"{'-' * 50}")
        
        sim = ThermoLLMSimulator(method=method, seed=Config.SEED)
        
        # Pre-train NS-PIRL agent
        if method == 'NS_PIRL':
            print("  Phase 1: Training D-DQN agent (5 episodes)...")
            for ep in range(5):
                train_sim = ThermoLLMSimulator(method='NS_PIRL', seed=Config.SEED + ep)
                train_sim.run(duration=1000, training=True)
                # Transfer learned weights
                sim.agent.q_network = train_sim.agent.q_network
                sim.agent.target_network = train_sim.agent.target_network
                sim.agent.epsilon = 0.1  # Reduce exploration
                print(f"    Episode {ep+1}/5 complete.")
            print("  Phase 2: Running evaluation...")
        
        summary = sim.run(training=False)
        results[method] = summary
        if method == 'NS_PIRL':
            trained_agent = sim.agent
        
        print(f"  Violations: {summary['violations']}")
        print(f"  Max Temperature: {summary['max_temperature']:.1f}°C")
        print(f"  STR: {summary['STR']:.1f} tokens/s")
        print(f"  Energy/token: {summary['energy_per_token']:.2f} J/token")
        
        if 'shield_vetoes' in summary:
            print(f"  Safety Shield vetoes: {summary['shield_vetoes']}")
            print(f"  Necessary Failures: {summary['necessary_failures']}")
    
    # Print comparison table (matching Table 3/4 of the paper)
    print("\n" + "=" * 70)
    print("  TABLE: PERFORMANCE COMPARISON (matching paper Table 3 & 4)")
    print("=" * 70)
    print(f"  {'Method':<16} {'Violations':>10} {'Max T(°C)':>10} {'STR(t/s)':>10} {'E(J/t)':>8}")
    print(f"  {'-' * 54}")
    
    for method in methods:
        r = results[method]
        str_improvement = ""
        if method == 'NS_PIRL' and 'RL_ONLY' in results:
            rl_str = results['RL_ONLY']['STR']
            if rl_str > 0:
                improvement = (r['STR'] - rl_str) / rl_str * 100
                str_improvement = f" (+{improvement:.0f}%)"
        
        print(f"  {method:<16} {r['violations']:>10} {r['max_temperature']:>10.1f} "
              f"{r['STR']:>10.1f}{str_improvement} {r['energy_per_token']:>8.2f}")
    
    # Save results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(output_dir, 'experiment_results.json')
    
    # Convert numpy types to native Python for JSON serialization
    json_results = {}
    for method, r in results.items():
        json_results[method] = {k: float(v) if isinstance(v, (np.floating, float)) 
                                else int(v) if isinstance(v, (np.integer, int))
                                else v for k, v in r.items()}
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")
    
    return results, trained_agent


def run_ablation_study(trained_agent=None):
    """
    Ablation study (Table 5 of the paper).
    Systematically removes one component to measure contribution.
    """
    print("\n" + "=" * 70)
    print("  ABLATION STUDY")
    print("=" * 70)
    
    configs = {
        'Full NS-PIRL': {'shield': True, 'twin': True, 'phase': True, 'boundary': True},
        'w/o Safety Shield': {'shield': False, 'twin': True, 'phase': True, 'boundary': True},
        'w/o Digital Twin': {'shield': True, 'twin': False, 'phase': True, 'boundary': True},
        'w/o Phase Observer': {'shield': True, 'twin': True, 'phase': False, 'boundary': True},
        'w/o Boundary Constraint': {'shield': True, 'twin': True, 'phase': True, 'boundary': False},
    }
    
    print(f"  {'Configuration':<30} {'Violations':>10} {'STR(t/s)':>10} {'E(J/t)':>8}")
    print(f"  {'-' * 58}")
    
    for name, cfg in configs.items():
        # Run full NS-PIRL for all configs (differences are in constraint enforcement)
        sim = ThermoLLMSimulator(method='NS_PIRL', seed=Config.SEED, ablation_cfg=cfg)
        
        if trained_agent is not None:
            sim.agent.q_network = trained_agent.q_network
            sim.agent.target_network = trained_agent.target_network
            sim.agent.epsilon = 0.1
            
        summary = sim.run(duration=2700, training=False)  # Shorter for ablation
        
        print(f"  {name:<30} {summary['violations']:>10} "
              f"{summary['STR']:>10.1f} {summary['energy_per_token']:>8.2f}")
    
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    print(f"\n  Thermo-LLM Framework v1.0")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results, trained_agent = run_all_experiments()
    run_ablation_study(trained_agent)
    
    print("\n" + "=" * 70)
    print("  EXPERIMENT COMPLETE")
    print("=" * 70)
    print("  All results saved. Ready for paper figures generation.")
    print("=" * 70)
