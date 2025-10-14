#!/usr/bin/env python3
"""
generate plots from pickle results
Usage: python generate_plots_from_pickle.py
"""

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import scipy.stats as st

# parameters
seed = 25092525
design_list = ['K Chain', 'Erdos-Renyi', 'Random Regular']
marker_list = {
    'Erdos-Renyi': 's', 
    'Random Regular': 'd',
    'K Chain': '*'
}
label_list = {
    'Erdos-Renyi': 'Erdős-Rényi',
    'Random Regular': r'Random $d$-Regular',
    'K Chain': r'$K$-Chain'
}
color_list = {
    'Erdos-Renyi': 'C0',
    'Random Regular': 'C3',
    'K Chain': 'C2'
}

def prob_binom_geq(n, d, p):
    """calculate the probability of binomial distribution P(X >= d)"""
    from scipy.stats import binom
    return 1 - binom.cdf(d-1, n, p)

def generate_plot_from_pickle(results, n, p, seed):
    """generate plots from pickle results"""
    
    # Create plots directory if it doesn't exist
    os.makedirs("results/plots", exist_ok=True)
    
    # calculate d_max based on the reference code logic
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
            plt.plot(d_list, loss_data, label=label_list[name], marker=marker_list[name], color=color_list[name])
            # Add confidence interval shading
            plt.fill_between(d_list, ci_low_data, ci_up_data, alpha=0.2, color=color_list[name])
    
    # Plot the lower bound with new formula
    lower_bound = []
    for d in d_list:
        q = 1 - p
        npq_d = n * p * (q ** d)
        
        # New formula: max{ npq^d - 1, 0.5*npq^d * (1 - (1-2p)^{n-d-1}) } 
        term1 = npq_d - (1-(1-2*p)**n)/2. 
        term2 = 0.5 * npq_d * (1 - ((1 - 2*p) ** (n - d - 1)))
        lower_val = max(term1, term2)
        lower_bound.append(lower_val)
    
    plt.plot(d_list, lower_bound, label=r'Lower Bound', linestyle='--', color='black')
    
    # Customize the plot
    plt.yscale("log")
    plt.xlabel("Average Degree", fontsize=14)
    plt.ylabel("Loss (log scale)", fontsize=14)
    plt.title(f"Performance for Middle-Mile Transportation (n={n}, p={p})", fontsize=14)
    
    plt.legend(fontsize=14, loc='lower left')
    plt.grid(True)
    
    # Save the plot
    plot_file = f"results/plots/loss_comparison_n_{n}_p_{p}.pdf"
    plt.tight_layout()
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {plot_file}")

def main():
    """主函数：处理所有pickle文件并生成图表"""
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
            generate_plot_from_pickle(results, n, p, seed)
            
        except Exception as e:
            print(f"Error processing {result_file}: {e}")

if __name__ == "__main__":
    main()
