import matplotlib.pyplot as plot
import networkx as nx
import numpy as np
import pandas as pd
import itertools

from genetic_algo import geneticAlgo
from proposedAlgo import proposedAlgo
from dag_generator import dag_generator

###
file_name = ""
# n_processors = 2
X = 0.5       #crossover rate
M = 0.1       #mutation rate
n_gen = 10 # number of generations
n_chroms = 10 # number of chromosomes

### parameters of the dag generator
# n_nodes       = 10   # total number of nodes
# width           = 0.5  # the width of the graph
# regular       = 1  # reflects the irregularity of the number of tasks per level around the perfect number
# density       = 0.5  # probability of creating an edge between nodes from different levels
jump          = 2  # how many possible levels between a parent and a child
min_comm_cost = 15    # minimum edge weight
max_comm_cost = 25   # maximum edge weight
min_duration  = 15   # minumum task execution time
max_duration  = 25   # maximum task execeution time


### simulation params
n_nodes = [10, 20, 30, 40, 60]
n_processors = [2, 4, 6]
width     = [0.5, 1, 10]   # the width of the graph
density = [0.1, 0.5, 1]   # probability of creating an edge between nodes from different levels
regular = [0.1, 0.5, 1]   # reflects the irregularity of the number of tasks per level around the perfect number
jump    = [2]        # how many possible levels between a parent and a child

columns = ['n_nodes', 'n_processors' ,'width', 'density', 'regular', 'jump', 'GA_makespan', 'proposedAlgo_makespan']
row  = dict(zip(columns, []))
rows = []
for i, values in enumerate(itertools.product(*[n_nodes, n_processors, width, density, regular, jump])) :
    row = dict(zip(columns, values))
    dag_object = dag_generator("generate_file.gml", n_nodes=row["n_nodes"], width=row["width"], regular=row["regular"], density=row["density"], min_comm_cost=min_comm_cost, max_comm_cost=max_comm_cost, min_duration=min_duration, max_duration=max_duration, jump=row["jump"])
    dag = dag_object.generate_dag()
    GA_object = geneticAlgo(dag, n_processors=row["n_processors"], n_generations = n_gen, n_chromosomes=n_chroms, X_rate = X, M_rate = M)
    GA_object.run()
    row['GA_makespan']  = GA_object.compute_makespan()
    proposedAlgo_object = proposedAlgo(dag, n_processors=row["n_processors"], r=0 )
    proposedAlgo_object.run()
    row['proposedAlgo'] = proposedAlgo_object.compute_makespan()
    #
    rows.append(row)
    if (i%20==0) :
        print("in progress, i = ", i)
    # print(rows)

# print(rows)
df = pd.DataFrame(rows)
df.to_csv("results_all.csv")
print(df)
