import matplotlib.pyplot as plot
import networkx as nx
import numpy as np
import pandas as pd
import itertools

from genetic_algo import geneticAlgo
from dag_generator import dag_generator

###
file_name = ""


### parameters of the dag generator
n_nodes       = 100   # total number of nodes
width           = 0.5  # the width of the graph
regular       = 1  # reflects the irregularity of the number of tasks per level around the perfect number
density       = 0.5  # probability of creating an edge between nodes from different levels
jump          = 2    # how many possible levels between a parent and a child
min_comm_cost = 1    # minimum edge weight
max_comm_cost = 20   # maximum edge weight
min_duration  = 1    # minumum task execution time
max_duration  = 20   # maximum task execeution time


### simulation params
n_processors = 4
X_rate = [0.1, 0.5, 1]
M_rate = [0.1, 0.5, 1]
n_generations = [5, 10,15]
n_chromosomes = [5, 10, 15]

columns = ['n_nodes', 'n_processors' ,'width', 'density', 'regular', 'jump', 'X_rate', 'M_rate', 'n_generations', 'n_chromosomes', 'best_generation', 'GA_makespan']
row  = dict(zip(columns, [n_nodes, n_processors, width, density, regular, jump]))
rows = []

# generate a DAG
dag_object = dag_generator("generated_file.gml", n_nodes=n_nodes, width=width, regular=regular, density=density, min_comm_cost=min_comm_cost, max_comm_cost=max_comm_cost, min_duration=min_duration, max_duration=max_duration, jump=jump)
dag = dag_object.generate_dag()

for i, values in enumerate(itertools.product(*[X_rate, M_rate, n_generations, n_chromosomes])) :
    row.update(dict(zip(['X_rate', 'M_rate', 'n_generations', 'n_chromosomes'], values)))

    GA_object = geneticAlgo(dag, n_processors=row["n_processors"], n_generations = row["n_generations"], n_chromosomes=row['n_chromosomes'], X_rate = row['X_rate'], M_rate = row['M_rate'])
    GA_object.run()
    row['GA_makespan']     = GA_object.compute_makespan()
    row['best_generation'] = GA_object.best_generation
    rows.append(row.copy())  # a copy is important, otherwise it will modify the previous values !!!
    if (i%10==0) :
        print("in progress, i = ", i)

# print(rows)
df = pd.DataFrame(rows)
df.to_csv("results_GA.csv")
print(df)
