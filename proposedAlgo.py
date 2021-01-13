import numpy as np
import networkx as nx


def gen_costs(L_tasks, n_processors, r):
    """
    """
    for t in L_tasks :
        task_duration = t[1]["duration"]
        t[1]["comp_cost"] = list(np.random.randint(np.ceil(task_duration*(1-r)), task_duration*(1+r)+1, size = n_processors))
    return list(L_tasks)

class proposedAlgo :
    def __init__(self, dag, n_processors, r = 0):
        """
        """
        self.dag     = dag
        self.tasks = gen_costs(self.dag.nodes(data=True), n_processors, r)  # initialy it has this form : [('T1', {'time': 2}), ('T2', {'time': 3}),....]
        # print(self.tasks)
        self.edges   = list(self.dag.edges(data=True))  # initialy it has this form : [('T1', 'T2', {'weight': 4}), ('T1', 'T3', {'weight': 1}),...]
        self.N_tasks = len(self.tasks)
        self.N_proc  = n_processors
        self.processors = [[] for i in range(n_processors)] # each list will contains the assigned tasks to the processor labeled by index
        self.makespan = 0
        # print("N° of tasks : {}\nN° of processors {}\n".format(self.N_tasks, self.N_proc))


    def run(self):
        """
        Run the different steps of the proposed algorithm
        """
        # compute weights
        self.compute_weights()
        #initialize tasks
        for i, task in enumerate(self.tasks) :
            task[1]["rank"]          = None
            task[1]["assigned_proc"] = None
            task[1]["start"]         = 0
            task[1]["finish"]        = 0
            # find the entry node
            if len(list(self.dag.predecessors(task[0]))) == 0:
                index_entry_node = i
        # rank the tasks
        self.ranking(self.tasks[index_entry_node][0])
        # print([t[0] for t in self.tasks])
        # sort tasks based on their ranking
        self.tasks.sort(key = lambda x: x[1]["rank"], reverse=True)
        # print([t[0] for t in self.tasks])
        # print([t[1]["rank"] for t in self.tasks])
        # generate a schedule
        self.schedule()


    def compute_weights(self):
        """
        Compute the weights required for the ranking step
        """
        for task in self.dag.nodes():
            high_weight = max(self.dag.nodes[task]['comp_cost'])
            low_weight  = min(self.dag.nodes[task]['comp_cost'])
            # print("high w = {}, low w = {}".format(high_weight, low_weight))
            if high_weight == 0 :
                self.dag.nodes[task]['weight'] = 0
            else :
                self.dag.nodes[task]['weight'] = low_weight*(high_weight - low_weight)/(high_weight)
        return 0


    def ranking(self, task):
        """
        Traverse the graph recursively in bottom-to-top topology and compute the ranks along the way.
        """
        proposed_rank = 0

        for successor in self.dag.successors(task):
            if self.dag.nodes[successor]["rank"] == None :
                self.ranking(successor)
            else :
                comm_cost = self.dag.get_edge_data(task, successor)["weight"]
                proposed_rank = max(self.dag.nodes[successor]["rank"] + comm_cost, proposed_rank)
        self.dag.nodes[task]['rank'] = self.dag.nodes[task]['weight'] + proposed_rank


    def compute_EFT(self, task, processor, comp_cost):
        """
        Compute the Earliest Finish Time of a task on the given processor
        """
        T_available = [] # availibility slots of this processor
        EST = 0
        #

        for predecessor in self.dag.predecessors(task):
                if self.dag.nodes[predecessor]["assigned_proc"] == processor :
                    comm_cost = 0  # assuming zero communication weight in the same processor
                else :
                    comm_cost = self.dag.get_edge_data(predecessor, task)["weight"]
                EST = max(EST, self.dag.nodes[predecessor]["finish"]+comm_cost)
        EFT = EST + comp_cost
        if self.processors[processor] == [] :
            return EFT
        else :
            for i, assigned_task in enumerate(self.processors[processor]):
                if i == 0:
                    if assigned_task[1]["start"] != 0: # if this task is not starting at time 0
                        T_available.append((0, assigned_task[1]["start"]))
                    else :
                        continue
                else :
                    T_available.append((self.processors[processor][i-1][1]["finish"], assigned_task[1]["start"]))
            T_available.append((self.processors[processor][-1][1]["finish"], np.inf))  # processor available after the execution of the last task
        for interval in T_available:
            if EST >= interval[0] and EFT <= interval[1]:
                return EFT
            elif EST < interval[0] and interval[0] + comp_cost <= interval[1]:
                return interval[0] + comp_cost

    def schedule(self):
        """
        Step 2 of the proposed algorithm, corresponding to the selection phase
        """
        # schedule the first task
        for i, task in enumerate(self.tasks):
            if len(list(self.dag.predecessors(task[0]))) == 0:
                index_entry_node = i
                weight_task0 = min(self.tasks[i][1]["comp_cost"])
                proc_task0   = (self.tasks[i][1]["comp_cost"]).index(weight_task0)
                self.tasks[i][1]["start"]         = 0
                self.tasks[i][1]["finish"]        = weight_task0
                self.tasks[i][1]["assigned_proc"] = proc_task0
                self.processors[proc_task0] += [self.tasks[i]]
                break

        # schedule the other tasks
        for task in self.tasks[:index_entry_node] + self.tasks[index_entry_node+1:] :
            min_EFT = np.inf
            for i, proc in enumerate(self.processors) :
                EFT = self.compute_EFT(task[0], i,  task[1]["comp_cost"][i])
                if EFT < min_EFT:
                    min_EFT = EFT
                    # select processor with min EFT
                    chosen_proc = i  # index corresponding to a processor

            w_fast_proc = min(task[1]["comp_cost"])
            fast_proc   = (task[1]["comp_cost"]).index(w_fast_proc)
            fast_EFT    = self.compute_EFT(task[0], fast_proc, w_fast_proc)

            if fast_EFT == min_EFT or fast_proc == chosen_proc: # global and optimal results are equal
                task[1]["start"]         = min_EFT - task[1]["comp_cost"][chosen_proc]
                task[1]["finish"]        = min_EFT
                task[1]["assigned_proc"] = chosen_proc
                self.processors[chosen_proc] += [task]
                self.processors[chosen_proc].sort(key = lambda t: t[1]["start"])
            else :
                # compute w_abstract and cross_threshold
                w_abstract = (fast_EFT-min_EFT)/(fast_EFT/min_EFT)
                cross_threshold = task[1]["weight"]/w_abstract
                if cross_threshold <= np.random.uniform(0.1, 0.3): # crossover
                    # map task to processor chosen_proc
                    task[1]["start"]         = min_EFT - task[1]["comp_cost"][chosen_proc]
                    task[1]["finish"]        = min_EFT
                    task[1]["assigned_proc"] = chosen_proc
                    self.processors[chosen_proc] += [task]
                    self.processors[chosen_proc].sort(key = lambda t: t[1]["start"])
                else :
                    task[1]["start"]         = fast_EFT - task[1]["comp_cost"][fast_proc]
                    task[1]["finish"]        = fast_EFT
                    task[1]["assigned_proc"] = fast_proc
                    self.processors[fast_proc] += [task]
                    self.processors[fast_proc].sort(key = lambda t: t[1]["start"])

    def compute_makespan(self):
        self.makespan = max([task[1]["finish"] for task in self.tasks])
        return self.makespan

    def __str__(self):
        """
        format the string that will be printed
        """
        output = ""
        output += "N° of tasks : {}\nN° of processors {}\n".format(self.N_tasks, self.N_proc)
        output += "Tasks are scheduled as follows : \n"
        for i, proc in enumerate(self.processors):
            output += "Processor {}:\n".format(i)
            for t in proc:
                output += "Task {} : start_time = {} finish_time = {}\n".format(t[0], t[1]["start"], t[1]["finish"])
        output += "Schedule length (makespan) = {} time units:\n".format(self.compute_makespan())
        return output


if __name__ == "__main__" :
    r = 0
    n_processors = 6
    dag = nx.read_gml("dag.gml")
    algo = proposedAlgo(dag, n_processors, r)
    algo.run()
    print(algo)
