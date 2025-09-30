import numpy as np
import time
import pickle
from flex_func import (
    FlexDesign_Eval,
    generate_scenarios,
    full_flex,
    k_chain,
    ER,
    prob_expander,
    random_regular
);

n_list = np.arange(100, 501, 100)
p_list = [0.8, 1/2, 1/4]
seed = 25092525

# Generate scenarios and save them 
#for n in n_list:
#    for p in p_list:
#        # generate 1000 scenarios for evaluation later
#        n_scenarios = 1000
#        supply_vec, demand_vec  = generate_scenarios(n_scenarios, n=n, p=p, #seed=seed)
#        scenarios = (supply_vec, demand_vec)
#
#        with open(f"results/scenarios/n_{n}_p_{p}_seed_{seed}.pkl", "wb") #as f:
#            pickle.dump(scenarios, f)

# Given n, p, and design, computed the expected loss for each d
def loss_per_d(n, design_name, e_model, full_evals, d_list, num_designs, seed):

    if design_name == 'K Chain':
        num_designs = 1

    final_results = {d: [] for d in d_list}

    for d in d_list:
        for i in range(num_designs): 
            if design_name == 'K Chain':
                design = k_chain(n, d)
            if design_name == 'Erdos-Renyi': 
                design = ER(n, d, seed + i*100)
            if design_name == 'Probabilistic Expander': 
                design = prob_expander(n, d, seed + i*100)
            if design_name == 'Random Regular': 
                design = random_regular(n, d, seed + i*100)
            e_model.update_design(design)
            evals = e_model.evaluate_expected_profit()

            final_results[d].append(full_evals - evals)

    return final_results

design_list = {
    'Erdos-Renyi': ER,
    'Probabilistic Expander': prob_expander,
    'Random Regular': random_regular,
    'K Chain': k_chain
}

num_designs = 10 # each random graph repeats this many times

def main():
    start = time.time()

    for n in n_list:
        for p in p_list:
            
            print('n', n, 'p', p)

            with open(f"results/scenarios/n_{n}_p_{p}_seed_{seed}.pkl", "rb") as f:
                supply_vec, demand_vec = pickle.load(f)
            
            d_max = int(np.ceil(np.log(1e-8/(n*p))/np.log(1-p)))
            d_list = list(range(2, min(d_max, 21), 2))
            
            final_results = {}
            
            # initial design to be zeros
            initial_design = np.zeros((n, n))
            e_model = FlexDesign_Eval(name="eval", demand_vec=demand_vec, supply_vec=supply_vec, design=            initial_design)
            
            # evaluate the full flex design as a benchmark
            design = full_flex(n)
            e_model.update_design(design)
            full_evals = e_model.evaluate_expected_profit()
            
            for design_name in design_list:
                print('design', design_name)
                # evaluate the design
                loss_results = loss_per_d(n, design_name, e_model, full_evals, d_list, num_designs, seed)
                print(f"n: {n}, p: {p}, {design_name}, loss results: {loss_results}")
                final_results[design_name] = loss_results
                print('time elapsed:', time.time() - start, 'seconds')
            
            with open(f"results/fix_n_p_results/n_{n}_p_{p}_seed_{seed}.pkl", "wb") as f:
                pickle.dump(final_results, f)
                            
if __name__ == "__main__":
    main()


