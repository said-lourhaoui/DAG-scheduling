# DAG-scheduling

#### The file "dag.gml" contain a random DAG with 20 nodes
#### To generate a schedule of the dag "dag.gml" using one of the two algorithms:   
- Make sure you have python 3.8 and above installed in your computer
- Make sure the following modules are already installed : networkx, numpy
- Place the script and the file "dag.gml" in the same folder.
- Run the desired script using and python IDE or by command line : "python3 genetic_algo.py" or "python3 proposedAlgo.py"
- The scripts will output a schedule and the makespan calcuted.


#### To generate a new DAG:  
- The following modules are required : numpy, networkx, matplotlib.pyplot
- Open the script dag_generator.py
- Specifiy the desired parameters
- Specify the file name where the dag will be saved
- Comment or uncomment the line "x.plot_dag(dag)", depending on whether you want to plot the graph or not
- Run the script

### To run the GA parameters estimation
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
- Finally, it will create a csv file containing the simulation results. 

### To visualize the GA paramaters estimation 
- Open the notebook (using jupyter notebook) Ga_plots.ipynb and run the cells

### To visualize the simulation for different parameters
- Open the noteboook (using jupyter notebook) Algorithms_comparison.ipynb and run  the cells
 
