#!/usr/bin/env python3
"""
only generate scenarios
"""

import numpy as np
import pickle
import os
from analyze_loss import generate_unipartite_scenarios

# current configuration
n_list = np.arange(600, 801, 200)  # [600, 800]
p_list = [0.25]
seed = 25092525

def generate_and_save_scenarios():
    """Generate and save node removal scenarios for all parameter combinations"""
    # Create directory if it doesn't exist
    os.makedirs("results/scenarios_uni", exist_ok=True)
    
    for n in n_list:
        for p in p_list:
            print(f"Generating node removal scenarios for n={n}, p={p}")
            # generate 1000 scenarios for evaluation later
            n_scenarios = 1000
            scenarios = generate_unipartite_scenarios(n_scenarios, n=n, p=p, seed=seed)
            
            filename = f"results/scenarios_uni/n_{n}_p_{p}_seed_{seed}.pkl"
            with open(filename, "wb") as f:
                pickle.dump(scenarios, f)
            print(f"Saved {len(scenarios)} scenarios to {filename}")

if __name__ == "__main__":
    print("Generating scenarios for n=600,800; p=0.25...")
    generate_and_save_scenarios()
    print("Scenarios generation completed!")
