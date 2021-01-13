###
#Author : Said Lourhaoui

###

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time

# width = 0.1
# n_nodes   = 100
# regular = 0.1
# density = 0.5
# jump = 2
# min_comm_cost = 0 # min edge weight
# max_comm_cost = 20 # max edge weight
# min_duration = 1   # min task execution time
# max_duration = 20

class dag_generator:
    def __init__(self, file_name, n_nodes = 10, width = 0.5, regular = 1, density =1, min_comm_cost=15, max_comm_cost =25, min_duration =15, max_duration=25, jump=2):
        self.file_name     = file_name
        self.n_nodes       = n_nodes
        self.width           = width
        self.regular       = regular
        self.density       = density
        self.min_comm_cost = min_comm_cost
        self.max_comm_cost = max_comm_cost
        self.min_duration  = min_duration
        self.max_duration  = max_duration
        self.jump = jump

    def generate_levels(self, n):
        # determine the perfect number of tasks per level according to width parameter
        nb_tasks_per_level = np.ceil((self.width*np.log(n)))

        tasks_per_level = [1] # start with one task for the first level
        total_nb_tasks = 1
        n_level = 1
        while total_nb_tasks < n :
            # assign a number of tasks to the current level depeding on the regalurity parameter
            new_level = np.random.randint(np.ceil(nb_tasks_per_level*self.regular), nb_tasks_per_level*(1+(1-self.regular))+1)
            if (total_nb_tasks + new_level > n) :
                new_level = n - total_nb_tasks

            tasks_per_level.append(new_level)
            n_level +=1
            total_nb_tasks += new_level
            #repeat until all tasks are distributed
        levels = [[] for level in range(n_level)]
        #update the list of levels
        m = 0
        for level, n_tasks_level in enumerate(tasks_per_level):
            levels[level] = [t+m for t in range(n_tasks_level)]
            m += n_tasks_level
        return levels

    def generate_adj_matrix(self, n, levels):
        n_levels = len(levels)
        adj_matrix = np.zeros((n,n))
        for i in range(1,n_levels):
            for j in range(len(levels[i])):
                # compute how many parents the task should have
                nb_parents = min(len(levels[i-1]),  1 + np.random.randint(0, self.density*len(levels[i-1])+1))
                #
                for k in range(nb_parents):
                    # get the level of the parent
                    p_level = (i- np.random.randint(1, self.jump+1))
                    p_level = max(0,p_level)
                    # randomly select which parent in p_level

                    p_index = np.random.randint(0, len(levels[p_level]))

                    parent = levels[p_level][p_index]
                    child  = levels[i][j]

                    adj_matrix[parent][child] = np.random.randint(self.min_comm_cost, self.max_comm_cost)
        return adj_matrix

    # generate a dictionary with random values (a value represents a task time)
    def generate_dict(self,n):
        """
        :param n: Number of keys (nodes in our case)
        :return:  A dictionary with n keys and random values (between a and b)
        """
        mydict = {}
        for i in range(0, n):
            mydict[i] = np.random.randint(self.min_duration, self.max_duration+1)
        return mydict

    def generate_dag(self):
        # determine levels
        levels = self.generate_levels(self.n_nodes)
        # generate the random adjacency matrix
        adj_matrix = self.generate_adj_matrix(self.n_nodes, levels)
        # determine start and end nodes if there exist many
        end_nodes = np.nonzero(np.all(adj_matrix==0, axis=1))[0]
        start_nodes = np.nonzero(np.all(adj_matrix==0, axis=0))[0]

        # convert the adjacency matrix to a directed acyclic graph
        dag = nx.from_numpy_matrix(adj_matrix, create_using=nx.DiGraph)
        # dag.edges(data=True)
        # add a node and merge with end nodes if there are many

        if len(end_nodes) > 1 :
            new_node = len(list(dag.nodes))
            for node in end_nodes :
                dag.add_edge(node, new_node, weight=np.random.randint(self.min_comm_cost, self.max_comm_cost+1))
        # add a node and merge with start nodes if there are many
        if len(start_nodes) > 1 :
            new_node = len(list(dag.nodes))
            for node in start_nodes :
                dag.add_edge(new_node, node, weight=np.random.randint(self.min_comm_cost, self.max_comm_cost+1))

        random_dict = self.generate_dict(len(dag.nodes()))
        ### set the value of each node which correspond to task duration
        nx.set_node_attributes(dag, random_dict, name='duration')
        return dag

    def write_dag(self, dag):
        nx.write_gml(dag, self.file_name)

    def plot_dag(self, dag):
        # draw the generated graph
        nx.draw_networkx(dag, with_labels=True)
        plt.draw()
        plt.show()

if __name__ == "__main__":
    # t0 = time.time()
    x = dag_generator("dag.gml", n_nodes = 20)
    dag = x.generate_dag()
    # print(time.time()-t0)
    x.plot_dag(dag)
    x.write_dag(dag)
