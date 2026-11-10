## Installation

Clone the repository:

```bash
git clone https://github.com/xniu01/random-regular-sparse-design.git
cd random-regular-sparse-design
```

Create and activate a Python environment (macOS/Linux):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies for the simulations and plotting:

```bash
python -m pip install numpy scipy pandas matplotlib networkx gurobipy jupyterlab
```

The process-flexibility experiments use Gurobi. Configure a Gurobi license that supports the experiment sizes; the size-limited license included with `gurobipy` is insufficient for the full experiments. See [Gurobi’s licensing instructions](https://support.gurobi.com/hc/en-us/articles/12684663118993-How-do-I-obtain-a-Gurobi-license).

## Process-flexibility experiments

Run these commands from the repository’s root directory.

To run a small example of the optimization model:

```bash
python flex_func.py
```

To run the experiment suite using the included demand scenarios:

```bash
python run_fix_n_p.py
```

The current configuration evaluates network sizes of 100, 200, 300, 400, and 500, with demand probabilities of 0.8, 0.5, and 0.25. Parameters are defined near the top of the script.

- Input scenarios: `results/scenarios/`
- Simulation outputs: `results/fix_n_p_results/`

To generate figures from saved results, open the plotting notebook and run its cells in order:

```bash
jupyter lab plot.ipynb
```

Figures are saved to `results/fix_n_p_results/plots/`. Saved results are included, so plotting does not require rerunning the optimization experiments.

## Transportation experiments

From the repository’s root directory:

```bash
cd tranport_matching
```

Run the experiments using the included scenarios:

```bash
python run_uni.py
```

The current configuration uses network sizes of 600 and 800 and a node-retention probability of 0.25.

To regenerate scenarios and then run the experiments:

```bash
python run_uni.py generate
```

To generate figures from existing results:

```bash
python run_uni.py plots
```

Within this folder:

- Input scenarios: `results/scenarios_uni/`
- Simulation outputs: `results/unipartite_results/`
- Figures: `results/plots/`

Full simulation runs evaluate many scenarios and graph designs. To explore a smaller configuration, adjust the parameters near the top of the corresponding script and ensure that matching scenario files are available.
