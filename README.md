# NEW DAG BASED SCHEDULING ALGORITHMS IN A MULTIPROCESSOR SYSTEM
The computation needs in various disciplines (Artificial intelligence, Quantum physics, Meteorology, etc.) 
have led to a very significant development of computers with multiprocessor architecture.
In this work, we are interested in solving the task assignment problem on a multiprocessor
architecture. We assume that the parallel tasks are identified and that the precedence constraints
between tasks are specified using a Directed Acyclic Graph. Two algorithms are
used, a proposed Genetic algorithm in [1] and a proposed algorithm in [2], they both try to
schedule tasks among processors while minimizing the makespan, which corresponds to
the finish of time of the last scheduled task.
This metric depends on the graph type, namely : number of nodes, the width, the density,
the regularity and other parameters. We have implemented a random DAG generator and
simulated both algorithms for different types of DAGs. The results show that the algorithm
proposed in [2] performs better in terms of the makespan, it is also faster than the proposed
Genetic algorithm [1] in terms of time complexity.

#### The file "dag.gml" contain a random DAG with 20 nodes
#### To generate a schedule of the dag "dag.gml" using one of the two algorithms:   
- Make sure you have python 3.8 and above installed in your computer
- Make sure the following modules are already installed : networkx, numpy
- Run the desired script from a python IDE or by command line : "python3 genetic_algo.py" or "python3 proposedAlgo.py"
- The scripts will output a schedule and the calcuted makespan


#### To generate a new DAG:  
- The following modules are required : numpy, networkx, matplotlib.pyplot
- Open the script dag_generator.py
- Specify the desired parameters
- Specify the file name where the dag will be saved
- Comment or uncomment the line "x.plot_dag(dag)", depending on whether you want to plot the graph or not
- Run the script

#### To run the GA parameters estimation
- The following modules are required : numpy, networkx, pandas, itertools, genetic_Algo, proposedAlgo, dag_generator 
- Open the script GA_run.py
- Tune the dag_generator parameters
- Run the script
- It will create a csv file containing the simulation results

#### To run the simulation for different parameters 
- The following modules are required : numpy, networkx, pandas, itertools, genetic_Algo, proposedAlgo, dag_generator 
- Open the script run_all
- Tune the GA parameters
- Choose the DAG parameters ranges
- Run the script
- It will select a tuple of values at each iteration and executes both algorithms
- Finally, it will create a csv file containing the simulation results

#### To visualize the GA paramaters estimation 
- Open the notebook Ga_plots.ipynb and run the cells

#### To visualize the simulation for different parameters
- Open the notebook Algorithms_comparison.ipynb and run  the cells

[1] Kaur, R. ,Singh, G.. ”Genetic algorithm solution for scheduling jobs in multiprocessor
environment.” India Conference (INDICON), 2012 Annual IEEE , IEEE, 2012
[2] S. AlEbrahim and I. Ahmad, “Task scheduling for heterogeneous computing systems,”
The Journal of Supercomputing, vol. 73, no. 6, 2313–2338, 2017
[3] Hunold Sascha, Casanova Henri and Suter Fr ´ed´ eric. (2011). ”From Simulation to Experiment:
A Case Study on Multiprocessor Task Scheduling”. IEEE International Symposium on
Parallel and Distributed ProcessingWorkshops and Phd Forum. 665-672. 10.1109/IPDPS.2011.201.
 
