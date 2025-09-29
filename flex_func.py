import numpy as np
from gurobipy import *

class FlexDesign_Eval:
    # Initialize a model by setting variables and constraints (with RHS values)
    # The model is a LP with the objective of maximizing profit
    def __init__(self, name, demand_vec, supply_vec, design):
        """
        :param name: Model name
        :param demand_vec: 2d vector where each row contains the demand vector of a scenario.
        :param supply_vec: 2d vector where each row contains the supply vector of a scenario.
        :param design: 2d binary matrix describing the network configuration design
        """
        self.m = Model(name)
        # define the model variables
        # f: the flow variable, cs: the supply plant constraints, ct: the demand product constraints
        self.f = {}
        self.cs = {}
        self.ct = {}
        self.ub = {}

        # self.demand_scenario_vec is a 2d vector where each row contains the demand vector of a scenario.
        # self.num_demand is the length of the demand vector in one scenario
        # self.num_supply is the length of the supply vector in one scenario
        self.num_scenario = len(demand_vec)
        self.num_demand = len(demand_vec[0])
        self.num_supply = len(supply_vec[0])

        # self.demand_scenario_vec is a 2d vector where each row contains the demand vector of a scenario.
        # self.supply_scenario_vec is a 2d vector where each row contains the supply vector of a scenario.
        self.demand_scenario_vec = demand_vec
        self.supply_scenario_vec = supply_vec
        
        for j in range(self.num_demand):
            for i in range(self.num_supply):
                # declare flow variable with upper bound
                self.f[i,j] = self.m.addVar(name='f_%s' % i + '%s' % j, ub=design[i,j]*10e30)
                # declare flow variable with upper bound and unit profit of 1
                self.f[i,j].setAttr(GRB.attr.Obj, 1)
        self.m.update()

        for i in range(self.num_supply):
            self.cs[i] = self.m.addConstr(quicksum(self.f[i, j] for j in range(self.num_demand)) <= 0, name='cs_%s' % i)
        for j in range(self.num_demand):
            self.ct[j] = self.m.addConstr(quicksum(self.f[i, j] for i in range(self.num_supply)) <= 0,
                                          name='ct_%s' % j)
        self.m.update()
        self.m.setAttr("ModelSense", GRB.MAXIMIZE)
        self.m.setParam('OutputFlag', 0)

    def solve_scenario(self, scenario_index):
        '''
        :param scenario_index: index of the scenario to be solved'''

        for i in range(self.num_supply):
            # set the right-hand side of the supply constraints
            self.cs[i].setAttr(GRB.attr.RHS, self.supply_scenario_vec[scenario_index][i])
        for j in range(self.num_demand):
            # set the right-hand side of the demand constraints
            self.ct[j].setAttr(GRB.attr.RHS, self.demand_scenario_vec[scenario_index][j])
        self.m.optimize()
        return self.m.objVal

    def evaluate_expected_profit(self):
        '''
        :return: expected profit of the model
        '''
        samp_profit = np.zeros(self.num_scenario)
        for s in range(self.num_scenario):
            samp_profit[s] = self.solve_scenario(s)
        return np.average(samp_profit)
    
    def update_design(self, design):
        '''
        :param design: new design matrix
        :return: None
        '''
        # update the design matrix
        for i in range(self.num_supply):
            for j in range(self.num_demand):
                self.f[i,j].setAttr(GRB.attr.UB, design[i,j]*10e30)
        self.m.update()

# Set supply vector to all ones, and demand vector where each entry is 1/p with prob. p and 0 otherwise.
def generate_scenarios(num_scenario, n, p, seed=None):
    '''
    :param num_scenario: number of scenarios
    :param n: number of supplies and demands (balanced case)
    :param p: probability of setting demand to 1/p
    :return: demand and capacity vector
    '''
    if seed is not None:
        np.random.seed(seed)
    # Generate random demand vector
    demand_vec = np.zeros((num_scenario, n))
    for i in range(num_scenario):
        for j in range(n):
            if np.random.rand() < p:
                demand_vec[i][j] = 1/p

    supply_vec = np.ones((num_scenario, n))
    return supply_vec, demand_vec

# Generate a full flexibility design for n plants and products.
def full_flex(n):
    
    E = np.ones((n, n), dtype=int)
    return np.array(E)

# Generate a K-chain design for n plants and products; k=d here.
def k_chain(n, d):
    k = d

    E = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(k):
            E[i][(i+j) % n] = 1
    return np.array(E)

# Generate an Erdos-Renyi design for n plants and products.
def ER(n, d, seed=None):

    if seed is not None:
        np.random.seed(seed)
    
    E = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(n):
            if np.random.rand() < d/n:
                E[i][j] = 1
    return np.array(E)

# Generate the probabilistic expander design [CZZ] for n plants and products.
def prob_expander(n, d, seed=None):
    Delta = int(np.floor(d/2))

    if seed is not None:
        np.random.seed(seed)
    
    E = np.zeros((n, n), dtype=int)

    # Step 1: For each plant i in U, uniformly sample Δ products with replacement to add edges
    for i in range(n):
        for _ in range(Delta):
            j = np.random.randint(0, n-1)
            E[i][j] = 1
    
    # Step 2: For each product j in W, uniformly sample Δ plants with replacement to add edges
    for j in range(n):
        for _ in range(Delta):
            i = np.random.randint(0, n-1)
            E[i][j] = 1

    return np.array(E)

# Generate a random d-regular bipartite graph.
def random_regular(n, d, seed=None):

    if seed is not None:
        np.random.seed(seed)
    
    E = np.zeros((n, n), dtype=int)
    # Step 1: For each supply, create d half-edges
    left_stubs = np.repeat(np.arange(n), d)
    # Step 2: For each demand, create d half-edges, and shuffle them
    right_stubs = np.repeat(np.arange(n), d)
    np.random.shuffle(right_stubs)
    # Step 3: Randomly match half-edges to create edges
    np.add.at(E, (left_stubs, right_stubs), 1)

    return np.array(E)

# Main function to test the FlexDesign_Evaluation class and generate_scenarios function
def main():
    # Example usage

    name = "Example1"
    # Example demand vector with 3 scenarios and 2 demands
    # Example supply vector with 2 scenarios and 2 supplies
    demand_vec = np.array([[10, 20], [30, 40], [50, 60]])
    print(f"demand vector scenario 1: {demand_vec[0]}")
    supply_vec = np.array([[10, 20], [30, 40], [50, 60]])
    design = np.array([[1, 0], [0, 1]])

    model = FlexDesign_Eval(name, demand_vec, supply_vec, design)
    profit = model.evaluate_expected_profit()
    print("Expected Profit:", profit)

    design2 = np.array([[0, 1], [1, 0]])
    model.update_design(design2)
    profit2 = model.evaluate_expected_profit()
    print("Expected Profit after design update:", profit2)

if __name__ == "__main__":
    main()