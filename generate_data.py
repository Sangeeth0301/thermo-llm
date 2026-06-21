import os
import csv
import random
import math

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_dataset():
    data_dir = os.path.join(OUTPUT_DIR, 'data')
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, 'd_thermo.csv')

    print(f"Generating telemetry dataset D_thermo...")
    print(f"Target location: {csv_path}")

    # Physical parameters
    T_amb = 28.0      # Ambient temperature (°C)
    R_th = 3.8        # Thermal resistance (°C/W)
    C_th = 75.0       # Thermal capacitance (J/°C)
    dt = 0.01         # 100 Hz sampling (10 ms intervals)
    
    # Models configuration: (prefill_power_factor, decode_power_factor)
    models = {
        'Gemma-2-9B': (7.5, 5.8),
        'Qwen-1.8B': (6.0, 4.5),
        'TinyLlama-1.1B': (4.2, 3.0)
    }
    
    # Frequencies supported (MHz)
    frequencies = [600, 900, 1200, 1500, 1800, 2400]

    # Initialize thermal state
    T_curr = T_amb + 2.0 # start slightly above ambient
    
    # CSV Headers
    headers = ['timestamp_sec', 'model_name', 'phase', 'frequency_mhz', 'power_w', 'junction_temp_c']
    
    total_steps = 20000  # 20,000 steps at 100 Hz = 200 seconds of telemetry data
    rows = []
    
    # Simulation state
    current_time = 0.0
    current_query = None
    query_time_remaining = 0.0
    query_phase = -1  # -1: Idle, 1: Prefill, 0: Decode
    active_model = 'Gemma-2-9B'
    active_freq = 2400
    
    for step in range(total_steps):
        # State machine for LLM workloads
        if query_time_remaining <= 0.0:
            if query_phase == -1: # Transition from Idle to Prefill
                active_model = random.choice(list(models.keys()))
                query_phase = 1 # Prefill
                query_time_remaining = random.uniform(0.5, 1.5) # Prefill duration (0.5 to 1.5s)
                active_freq = 2400 # Default to high freq for prefill
            elif query_phase == 1: # Transition from Prefill to Decode
                query_phase = 0 # Decode
                query_time_remaining = random.uniform(5.0, 12.0) # Decode duration (5 to 12s)
                # D-DQN frequency scaling selection simulation during decode
                # Scale frequency down if temperature is getting high
                if T_curr > 41.5:
                    active_freq = random.choice([900, 1200])
                elif T_curr > 38.0:
                    active_freq = random.choice([1500, 1800])
                else:
                    active_freq = 2400
            elif query_phase == 0: # Transition from Decode to Idle
                query_phase = -1 # Idle
                query_time_remaining = random.uniform(2.0, 4.0) # Idle duration (2 to 4s)
                active_freq = 600 # Scale down to idle frequency
        
        # Calculate power draw (P_in) based on state, model and frequency
        if query_phase == 1: # Prefill (compute-bound, high power)
            p_factor = models[active_model][0]
            # Power scales with frequency cubed (dynamic) + leakage (static)
            freq_ratio = active_freq / 2400.0
            power = p_factor * (0.8 * (freq_ratio ** 3) + 0.2 * freq_ratio)
        elif query_phase == 0: # Decode (memory-bandwidth-bound, medium power)
            p_factor = models[active_model][1]
            freq_ratio = active_freq / 2400.0
            power = p_factor * (0.7 * (freq_ratio ** 2) + 0.3 * freq_ratio)
        else: # Idle
            freq_ratio = active_freq / 2400.0
            power = 0.25 * freq_ratio + 0.05
            
        # Add small power noise
        power += random.normalvariate(0, 0.08)
        power = max(0.1, power) # clamp to positive
        
        # Integrate temperature (dT/dt) using physical heat diffusion equations (First-order RC equivalent)
        # dT = (dt / C_th) * (P_in - (T_curr - T_amb)/R_th)
        dT = (dt / C_th) * (power - (T_curr - T_amb) / R_th)
        # Add small thermal process noise
        thermal_noise = random.normalvariate(0, 0.002)
        T_curr += dT + thermal_noise
        
        # Clamp physically
        T_curr = max(T_amb, T_curr)
        
        # Save record
        rows.append([
            round(current_time, 2),
            active_model if query_phase != -1 else 'Idle',
            query_phase,
            active_freq,
            round(power, 3),
            round(T_curr, 3)
        ])
        
        # Update trackers
        current_time += dt
        query_time_remaining -= dt

    # Write to CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Successfully generated {len(rows)} telemetry rows.")
    print("Dataset variables: timestamp_sec, model_name, phase (-1:idle, 0:decode, 1:prefill), frequency_mhz, power_w, junction_temp_c")

if __name__ == '__main__':
    generate_dataset()
