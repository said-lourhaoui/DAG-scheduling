import networkx as nx
import numpy as np
import copy

class Task:
    def __init__(self,id, duration):
        self.id       = id
        self.duration = duration
        self.assigned_proc = None
        self.start         = 0
        self.finish        = 0
        self.min_completion_time  = -1

class population:
    def __init__(self, dag, n_processors, N_p):
        self.dag = dag
        self.n_processors = n_processors
        self.n_chromosomes = N_p
        self.random_pop    = None

    def generate_chromosome(self):
        chromosome_tasks = []
        for node in list(self.dag.nodes(data=True)):
            chromosome_tasks.append(Task(node[0], node[1]["duration"]))
        chromosome_tasks = np.random.choice(chromosome_tasks, size = len(chromosome_tasks), replace = False)
        chromosome_processors = np.random.choice(range(self.n_processors), size = len(chromosome_tasks))

        chromosome = np.array([chromosome_tasks, chromosome_processors, 0], dtype=object)# put the first array and second in a row corresponding to a chromosome,
        #the third elelement correspond to the fitness, initialy set to 0
        return chromosome

    def gen_random_population(self):
        pop = []
        for i in range(self.n_chromosomes):
            pop.append(self.generate_chromosome())
        self.random_pop = np.array(pop)
        return self.random_pop



class geneticAlgo :
    def __init__(self, dag, n_processors, n_generations, n_chromosomes, X_rate, M_rate):
        """
        """
        self.dag = dag
        self.tasks = np.array(self.dag.nodes(data=True) , dtype = ('object,object'))
        self.N_tasks = len(self.tasks)
        self.N_generations = n_generations
        self.N_proc = n_processors
        self.processors_schedules = [[] for i in range(n_processors)] # each list will contains the assigned tasks to the processor labeled by index
        self.p_size    = (n_chromosomes, self.N_tasks) # population size ( number of chromosomes x number of tasks)
        self.X_rate    = X_rate # crossover rate
        self.M_rate    = M_rate # mutation rate
        self.current_pop   = population(self.dag, n_processors, n_chromosomes).gen_random_population()
        self.fittest       = self.current_pop[0] # inially we consider this chromosome is the best
        self.best_generation    = 1 # generation of the best solution found

    def initialize_tasks(self, population):
        for chromosome in population :
            for task in chromosome[0] :
                task.assigned_proc = None
                task.start         = 0
                task.finish        = 0
                # task.min_completion_time"] = -1
            # for task in self.dag.nodes() :
                self.compute_min_completion_time(task)

    def compute_min_completion_time(self, task):
        """
        Computes the minimum completion time of this task based on the minimum completion times of its dependencies
        """
        if task.min_completion_time < 0:
            task.min_completion_time = task.duration
            if len(list(self.dag.predecessors(task.id)))>0:
                task.min_completion_time += max([self.compute_min_completion_time(Task(predecessor,self.dag.nodes[predecessor]["duration"])) for predecessor in self.dag.predecessors(task.id)])
        return task.min_completion_time


    def crossover(self, chromosome1, chromosome2):
        """
        This function randomly applies one of the two following crossover operators :
        - single point crossover : This operator is applied to the 2nd array of chromosomes
        - proposed crossover :  This operator is applied to the 1st array of chromosomes
        """
        ch1_array1, ch1_array2, ch1_fit = chromosome1[0], chromosome1[1], chromosome1[2]
        ch2_array1, ch2_array2, ch2_fit = chromosome2[0], chromosome2[1], chromosome2[2]
        if np.random.random() < self.X_rate :
            if np.random.random() < 0.5 : # single point crossover
                # select a crossover point randomly between 1 to number of tasks
                crossover_pt = np.random.randint(1, self.N_tasks)
                # The portions of chromosomes lying to the right of crossover_pt point are
                #exchanged to produce two offsprings
                offspring1_array2   = np.concatenate(([p for p in ch1_array2[:crossover_pt]], [p for p in ch2_array2[crossover_pt:]]), axis=None)
                offspring2_array2   = np.concatenate(([p for p in ch2_array2[:crossover_pt]], [p for p in ch1_array2[crossover_pt:]]), axis=None)
                # fill the two offstrings
                offspring1 = np.array([ch1_array1, offspring1_array2, ch1_fit], dtype=object)
                offspring2 = np.array([ch2_array1, offspring2_array2, ch2_fit], dtype=object)
                return offspring1, offspring2

            else :      # proposed crossover
                # select a crossover point randomly between 1 to number of tasks
                crossover_pt = np.random.randint(1, self.N_tasks)
                ch1_array1_ids = [t.id for t in ch1_array1]
                # selected elements of the part in the left of the crossover point
                left_tasks = [Task(t.id, t.duration) for t in ch1_array1[:crossover_pt] ]
                # selected elements of the part in the right of the crossover point
                right_tasks = [Task(t.id, t.duration) for t in ch2_array1 if t.id not in ch1_array1_ids[:crossover_pt] ]
                # list of tasks of the offspring
                offspring_array1 = np.concatenate((left_tasks, right_tasks), axis=0)
                # offspring_array1 = ch1_array1[:crossover_pt] + L
                offspring        = np.array([offspring_array1, ch1_array2, ch1_fit], dtype=object)
                return offspring, chromosome2
        else : # crossover is not applied
            return chromosome1, chromosome2


    def mutation(self, chromosome):
        """
        applies the mutation operator to the 2nd array of the chromosome
        """
        array1, array2, ch_fit = chromosome[0], chromosome[1] , chromosome[2]
        # Ntasks = len(array1)
        for i in range(self.N_tasks):
            # Each element in the second array of the chromosome is subjected to mutation with a probability M_rate
            if np.random.random() < self.M_rate :
                # randomly assign another processor to this task
                array2[i] = np.random.choice(list(array2[:i]) + list(array2[i+1:]))
        offspring        = np.array([array1, array2, ch_fit], dtype=object)
        return offspring

    def selection(self, population):
        """
        select two chromosomes of the population using roulette wheel procedure
        """
        fitnesses = population[:,2]
        # calculate sum of all chromosomes fitnesses
        S = sum(fitnesses)
        # generate random number r from interval (0,S)
        r = np.random.uniform(0,S)
        # go through the population and sum fitnesses from 0 - S
        # when the partial_sum s is greater than r, stop and return the current chromosome
        partial_sum = 0
        for i in range(len(population)):
            partial_sum += population[i][2]
            if partial_sum >= r :
                #return i
                return population[i-1], population[i]
        return -1

    def fitness(self, population):
        """
        Compute and store the fitness of each (solution) chromosome
        """
        updated_pop = []
        for chromosome in population :
            # get the schedule length
            last_task_scheduled  = max(chromosome[0], key= lambda task:task.finish)
            schedule_length = last_task_scheduled.finish
            # fill the the 3rd column
            chromosome[2] = 1/schedule_length
            updated_pop.append(chromosome)
        return np.array(updated_pop)

    def heuristic(self):
        """
        Apply the heuristic decoding to a chromosome to generate solution for each chromosome
        """
        solution = []
        for chromosome in self.current_pop :
            # Build a task list from a chromosome
            Ready_list  = list(chromosome[0])
            #
            Processor_allocations = chromosome[1]

            # Generate first schedule based on min completion time
            Ready_list.sort(key=lambda task: task.min_completion_time)
            # Task from the ready list is selected and scheduled to the available
            # processor on which the start time of the task is the earliest
            processors_schedules = [[] for i in range(self.N_proc)]
            for i, task in enumerate(Ready_list) :
                # Initially !!!!
                task.assigned_proc = Processor_allocations[i]
                processors_schedules[Processor_allocations[i]].append(task)

            # compute start and finish time of each task for the current chromosome
            for i, task in enumerate(Ready_list) :
                if len(list(self.dag.predecessors(task.id))) == 0: # if this task does not have a procedessor, it is scheduled first
                    task.start  = 0
                    task.finish = task.duration
                else :
                    predecessors_finish_time = []  #
                    for predecessor in self.dag.predecessors(task.id) :
                        # find the predecessor task in the ready_list
                        for t in Ready_list :
                            if t.id == predecessor :
                                break
                        # if task and its predecessor are assigned to the same processor, communication weight is zero
                        if task.assigned_proc == t.assigned_proc:
                            predecessors_finish_time.append(t.finish)
                        else : # if the they are assigned to different processors, communication weight will be included
                            comm_cost = self.dag.get_edge_data(predecessor, task.id)["weight"]
                            predecessors_finish_time.append(t.finish + comm_cost)
                    # now we have the list of predecessors finish time, we take the maximum
                    task.start  = max(predecessors_finish_time)
                    task.finish = task.start + task.duration
                # ordering of the tasks within the same processor
                for i, processor in enumerate(processors_schedules):
                    processor.sort(key = lambda t : t.start)
                    for j, task in enumerate(processor):
                        if j > 0 :
                            task.start = max(task.start, max(processor[:j], key = lambda t : t.finish).finish)
                            task.finish = task.start + task.duration
            # update the population
            #solution.append(chromosome)
        #return np.array(solution)



    def run(self):
        """
        This function executes the steps of the proposed genetic algorithm
        - heuristic decoding
        - fitness computation
        - selection
        - crossover and mutation
        """
        # print(self.current_pop)
        for i in range(self.N_generations):
            ## initialize tasks attributes
            self.initialize_tasks(self.current_pop)
            ##
            self.heuristic()
            ##
            self.current_pop = self.fitness(self.current_pop)
            # sort the population based on the fitness value
            self.current_pop = self.current_pop[np.argsort(self.current_pop[:,2])[::-1]]
            # update the fittest solution
            if self.fittest[2] < self.current_pop[0][2] : # if we find a better fitness we save the chromosme
                self.fittest = copy.deepcopy(self.current_pop[0])
                self.best_generation = i + 1
            # save the two fittest chromosomes as parents to contribute in the next generation
            next_generation = self.current_pop[0:2]
            # apply the operators to reproduce
            for j in range(int(len(self.current_pop) / 2) - 1):
                parents = self.selection(self.current_pop)
                offspring_1, offspring_2 = self.crossover(parents[0], parents[1])
                offspring_1 = self.mutation(offspring_1)
                offspring_2 = self.mutation(offspring_2)
                next_generation = np.append(next_generation, [offspring_1, offspring_2], axis=0)
            # update the current population
            self.current_pop = next_generation

        # sort the current pop
        self.current_pop = self.current_pop[np.argsort(self.current_pop[:,2])[::-1]]
        return self.current_pop

    def compute_makespan(self):
        self.makespan = max([task.finish for task in self.fittest[0]])
        return self.makespan

    def __str__(self):
        """
        format the string that will be printed
        """
        output = ""
        output += "N° of tasks : {}\nN° of processors {}\n".format(self.N_tasks, self.N_proc)
        output += "Tasks are scheduled as follows : \n"
        # fittest_chromosome = GAalgo.fittest[0]
        for i, task in enumerate(GAalgo.fittest[0]):
            self.processors_schedules[task.assigned_proc].append(task)

        for i, proc in enumerate(self.processors_schedules):
            output += "Processor {}:\n".format(i)
            for t in proc:
                output += "Task {} : start_time = {} finish_time = {}\n".format(t.id, t.start, t.finish)
        output += "Schedule length (makespan) = {} time units:\n".format(self.compute_makespan())
        return output

if __name__ == "__main__" :
    n_processors = 6
    n_generations = 10
    dag = nx.read_gml("dag.gml")
    GAalgo = geneticAlgo(dag, n_processors, n_generations, n_chromosomes=10, X_rate = 0.5, M_rate = 0.1)
    GAalgo.run()

    print(GAalgo)
