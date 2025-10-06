import numpy as np
import time
import pickle
import os
import matplotlib.pyplot as plt
from analyze_loss import (
    generate_unipartite_scenarios,
    complete_graph,
    random_regular_unipartite,
    ER_unipartite,
    k_chain_unipartite,
    cost_function
);

n_list = np.arange(100, 501, 100)
p_list = [0.8, 0.5, 0.25]  # probability of retaining each node
seed = 25092525

# Generate node removal scenarios and save them 
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
            
            with open(f"results/scenarios_uni/n_{n}_p_{p}_seed_{seed}.pkl", "wb") as f:
                pickle.dump(scenarios, f)
            print(f"Saved {len(scenarios)} scenarios for n={n}, p={p}")

# Given n, p, scenarios, and graph type, compute the expected loss for each d using node removal model
def loss_per_d_unipartite(n, graph_type, scenarios, complete_expected_cost, d_list, num_designs, seed, p):

    if graph_type == 'K Chain':
        num_designs = 1

    final_results = {d: [] for d in d_list}

    for d in d_list:
        losses_for_this_d = []  # Collect all losses for this degree
        
        for i in range(num_designs): 
            try:
                # Generate graph based on type
                if graph_type == 'Random Regular':
                    graph_adj = random_regular_unipartite(n, d, seed + i*100)
                elif graph_type == 'Erdos-Renyi':
                    graph_adj = ER_unipartite(n, d, seed + i*100)
                elif graph_type == 'K Chain':
                    graph_adj = k_chain_unipartite(n, d, seed + i*100)
                else:
                    raise ValueError(f"Unknown graph type: {graph_type}")
                
                # Calculate expected matching value for this graph across scenarios using node removal model
                graph_matching_values = []
                for scenario in scenarios:
                    matching_value = cost_function(graph_adj, scenario)
                    graph_matching_values.append(matching_value)
                
                graph_expected_matching = np.mean(graph_matching_values)
                
                # Calculate loss for this graph (complete matching - graph matching)
                loss = 2 * (complete_expected_cost - graph_expected_matching)
                losses_for_this_d.append(loss)
                
            except ValueError:
                # Skip if graph generation fails
                continue
        
        # Calculate average loss across all designs for this degree
        if losses_for_this_d:
            avg_loss = np.mean(losses_for_this_d)
            std_loss = np.std(losses_for_this_d) if len(losses_for_this_d) > 1 else 0.0
            final_results[d] = {'avg_loss': avg_loss, 'std_loss': std_loss, 'num_designs': len(losses_for_this_d)}
        else:
            final_results[d] = {'avg_loss': 0.0, 'std_loss': 0.0, 'num_designs': 0}

    return final_results

graph_type_list = [
    'Random Regular',
    'Erdos-Renyi', 
    'K Chain'
]

num_designs = 10 # each random graph repeats this many times

def generate_plot(results, n, p, seed):
    """Generate and save a plot for the results"""
    import scipy.stats as st
    
    # Create plots directory if it doesn't exist
    os.makedirs("results/plots", exist_ok=True)
    
    # Extract data for plotting
    design_list = ['Random Regular', 'Erdos-Renyi', 'K Chain']
    marker_list = {
        'Random Regular': '+',         # 加号
        'Erdos-Renyi': 'x',           # X号
        'K Chain': 's'                # 方块
    }
    
    # Calculate d_max based on the reference code logic
    d_max_1 = int(np.ceil(np.log(1e-5/(n*p))/np.log(1-p))) if p < 1 else 20
    d_max_2 = d_max_1
    d_list_1 = list(range(2, min(d_max_1, 21), 2))
    
    # Find where curves reach 0 to stop plotting
    for name in design_list:
        if name in results:
            for d in d_list_1:
                if d in results[name] and results[name][d]['avg_loss'] == 0:
                    d_max_2 = min(d, d_max_2)
                    break
    
    d_list = list(range(2, min(d_max_1, d_max_2, 21), 2))
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    
    # Plot each graph type
    for name in design_list:
        if name in results:
            loss_data = []
            ci_low_data = []
            ci_up_data = []
            
            for d in d_list:
                if d in results[name]:
                    res_mean = results[name][d]['avg_loss']
                    res_std = results[name][d]['std_loss']
                    num_designs = results[name][d]['num_designs']
                    
                    if name != 'K Chain' and num_designs > 1:
                        # Calculate confidence interval for non-K Chain graphs
                        res_st = res_std / np.sqrt(num_designs)  # Standard error of mean
                        ci_low, ci_up = st.t.interval(0.95, num_designs-1, loc=res_mean, scale=res_st)
                    else:
                        ci_low, ci_up = res_mean, res_mean
                    
                    loss_data.append(res_mean)
                    ci_low_data.append(ci_low)
                    ci_up_data.append(ci_up)
                else:
                    loss_data.append(0)
                    ci_low_data.append(0)
                    ci_up_data.append(0)
            
            # Plot the main line
            plt.plot(d_list, loss_data, label=name, marker=marker_list[name])
            # Add confidence interval shading
            plt.fill_between(d_list, ci_low_data, ci_up_data, alpha=0.2)
    
    # Plot the lower bound with new formula
    lower_bound = []
    for d in d_list:
        q = 1 - p
        npq_d = n * p * (q ** d)
        
        # New formula: max{ npq^d - 1, 0.5*npq^d * (1 - (1-2p)^{n-d-1}) }
        term1 = npq_d - 1
        term2 = 0.5 * npq_d * (1 - ((1 - 2*p) ** (n - d - 1)))
        lower_val = max(term1, term2)
        lower_bound.append(lower_val)
    
    plt.plot(d_list, lower_bound, label=r'Lower Bound', linestyle='--', color='black')
    
    # Customize the plot
    plt.yscale("log")
    plt.xlabel("Degree")
    plt.ylabel("Loss vs Full Flexibility (log scale)")
    plt.title(f"n={n}, p={p}")
    
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plot_file = f"results/plots/loss_comparison_n_{n}_p_{p}.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {plot_file}")

def main():
    start = time.time()
    
    # Create results directory if it doesn't exist
    os.makedirs("results/unipartite_results", exist_ok=True)

    for n in n_list:
        for p in p_list:
            
            print('n', n, 'p', p)

            # Check if scenarios file exists
            scenarios_file = f"results/scenarios_uni/n_{n}_p_{p}_seed_{seed}.pkl"
            if not os.path.exists(scenarios_file):
                print(f"ERROR: Scenarios file not found: {scenarios_file}")
                print("Please run generate_and_save_scenarios() first")
                continue

            # Load node removal scenarios
            try:
                with open(scenarios_file, "rb") as f:
                    scenarios = pickle.load(f)
                print(f"Loaded {len(scenarios)} scenarios")
            except Exception as e:
                print(f"ERROR: Failed to load scenarios - {e}")
                continue
            
            # Calculate d_max based on unipartite graph constraints
            # For unipartite graphs, we need d*n to be even for d-regular graphs to exist
            d_max = min(20, n-1)  # Reasonable upper bound for degree
            d_list = [d for d in range(2, d_max+1, 2) if (d * n) % 2 == 0]  # Only even degrees where d*n is even
            print(f"Testing degrees: {d_list}")
            
            final_results = {}
            
            # Calculate complete graph expected matching value as benchmark using node removal model
            print("Calculating complete graph benchmark...")
            complete_adj = complete_graph(n)
            complete_matching_values = []
            for i, scenario in enumerate(scenarios):
                matching_value = cost_function(complete_adj, scenario)
                complete_matching_values.append(matching_value)
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(scenarios)} scenarios")
            complete_expected_cost = np.mean(complete_matching_values)
            print(f"Complete graph expected matching value: {complete_expected_cost:.4f}")
            
            # Calculate theoretical lower bound for node removal scenarios
            theoretical_lower_bound = []
            for d in d_list:
                # Formula: n*p*(1-p)^d
                lower_bound_value = n * p * ((1 - p) ** d)
                theoretical_lower_bound.append(lower_bound_value)
            print(f"Theoretical lower bound calculated for {len(d_list)} degrees")
            
            for graph_type in graph_type_list:
                print(f'Evaluating graph_type: {graph_type}')
                # evaluate the graph type
                loss_results = loss_per_d_unipartite(n, graph_type, scenarios, complete_expected_cost, d_list, num_designs, seed, p)
                
                # Display results in a more readable format
                print(f"n: {n}, p: {p}, {graph_type} results:")
                for d, result in loss_results.items():
                    print(f"  d={d}: avg_loss={result['avg_loss']:.4f}±{result['std_loss']:.4f} (from {result['num_designs']} designs)")
                
                final_results[graph_type] = loss_results
                print('time elapsed:', time.time() - start, 'seconds')
            
            # Add theoretical lower bound to results
            final_results['theoretical_lower_bound'] = {
                'degrees': d_list,
                'lower_bound': theoretical_lower_bound
            }
            
            # Save results
            results_file = f"results/unipartite_results/n_{n}_p_{p}_seed_{seed}.pkl"
            try:
                with open(results_file, "wb") as f:
                    pickle.dump(final_results, f)
                print(f"Saved results to {results_file}")
            except Exception as e:
                print(f"ERROR: Failed to save results - {e}")
            
            # Generate and save plot
            try:
                generate_plot(final_results, n, p, seed)
                print(f"Generated plot for n={n}, p={p}")
            except Exception as e:
                print(f"ERROR: Failed to generate plot - {e}")

def generate_plots_for_existing_results():
    """Generate plots for all existing result files"""
    results_dir = "results/unipartite_results"
    if not os.path.exists(results_dir):
        print("No results directory found")
        return
    
    # Find all result files
    result_files = [f for f in os.listdir(results_dir) if f.endswith('.pkl')]
    print(f"Found {len(result_files)} result files")
    
    for result_file in result_files:
        try:
            # Parse filename to extract parameters
            parts = result_file.replace('.pkl', '').split('_')
            n = int(parts[1])
            p = float(parts[3])
            
            print(f"Generating plot for n={n}, p={p}")
            
            # Load results
            with open(os.path.join(results_dir, result_file), 'rb') as f:
                results = pickle.load(f)
            
            # Generate plot
            generate_plot(results, n, p, seed)
            
        except Exception as e:
            print(f"Error processing {result_file}: {e}")

def run_with_scenario_generation():
    """Run the full pipeline including scenario generation"""
    print("Step 1: Generating scenarios...")
    generate_and_save_scenarios()
    
    print("\nStep 2: Running main analysis...")
    main()
                            
if __name__ == "__main__":
    # You can choose to run either:
    # 1. Just main() if scenarios are already generated
    # 2. run_with_scenario_generation() to generate scenarios first
    # 3. generate_plots_for_existing_results() to generate plots for existing results
    
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "generate":
            print("Running with scenario generation...")
            run_with_scenario_generation()
        elif sys.argv[1] == "plots":
            print("Generating plots for existing results...")
            generate_plots_for_existing_results()
        else:
            print("Unknown argument. Use 'generate' or 'plots'")
    else:
        print("Running main analysis (assuming scenarios exist)...")
        print("Use 'python run_uni.py generate' to generate scenarios first")
        print("Use 'python run_uni.py plots' to generate plots for existing results")
        main()