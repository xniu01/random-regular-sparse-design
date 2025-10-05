# Graph Matching Loss Analysis

This is the code for tansportation flexibility, to verify the optimality of random regular graphs.

The file `analyze_loss.py` defines functions for the optimization problem and for generating different graph designs. The file `run_uni.py` runs those functions and gets the results.

Here num_scenarios = 1000, num_designs = 10, n_list = [100, 200, 300, 400, 500], and p_list = [0.8, 0.5, 0.25].

The plots are in `results/plots/`.

## Usage

```bash
python run_uni.py
```

## Dependencies

```bash
pip install numpy networkx matplotlib scipy
```
