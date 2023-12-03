import numpy as np
# you need to install this package `ioh`. Please see documentations here:
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass
from typing import Union, Optional
from datetime import datetime

budget = 5000
dimension = 50

# To make your results reproducible (not required by the assignment),
# you could set the random seed by
np.random.seed(3366766)

# %% Define the variator and selection functions


def mutate(population: np.ndarray[bool],
           p_mutation: Union[np.ndarray[float], None] = None):
    """
    Mutate the given population with gene-wise probability p_mutation.

    Takes the population and alters each gene of each individual with
    probability p_mutation; takes place in-place, so as to avoid having to
    copy over the full population every generation.

    Parameters
    ----------
    population : np.ndarray[bool]
        Population expressed as Boolean array with each row corresponding
        to an individual and each column to a gene.
    p_mutation : np.ndarray of float or NoneType, optional
        Probability of mutation per gene. If set to None, the "standard" value
        of 1/population.shape[1] is used, which flips on average one gene
        per individual. Also supports array values, in which case all values
        must be float and the dimension must match . The default is None.

    Returns
    -------
    population : np.ndarray[bool]
        Mutated population; the operation occurs in-place to avoid having to
        copy over the full population every generation.
    """
    # If p_mutation is not set, set it equal to the gene length
    if p_mutation is None:
        p_mutation = 1/population.shape[1]
    # (Randomly) determine which genes are mutated
    mutate_array = np.where(np.random.uniform(0, 1, population.shape) <
                            p_mutation[:, None], True, False)
    # Flip the selected bits in-place using array masking
    # We use the fact that Python uses 0 and 1 interchangeably with False and
    # True to do this quickly.
    population[mutate_array] = 1 - population[mutate_array]
    return population


def crossover(population: np.ndarray[bool],
              n_crossover: Union[int, str],
              p_crossover: Union[float, np.ndarray[float]]):
    """
    Perform cross-over on the population in place.

    Perform in-place cross-over on the population. Supports arbitrarily many
    cross-over points (so long as it is less than the genome length) and
    uniform crossover.

    [1] de Jong, K. A. (1975). Analysis of the behavior of a class of
        genetic adaptive systems.
    [2] Schaffer, J. D., Caruana, R. A., Eshelman, L. J., & Das, R. (1989).
        A Study of Control Parameters Affecting Online Performance of
        Genetic Algorithms for Function Optimization. Proceedings of the
        Third International Conference on Genetic Algorithms, 51–60.
        https://www.researchgate.net/publication/220885653

    Parameters
    ----------
    population : np.ndarray[bool]
        Population on which to perform cross-over.
    n_crossover : int or str
        Number of crossover points. Supports values of 1 up to
        population.shape[1], and "uniform", which applies uniform crossover.
        If "uniform", p_crossover is the crossover probability of individual
        genes. If int, the crossover locations are chosen at random.
    p_crossover : float or np.ndarray of floats
        Probability that a given sequential pair undergoes crossover.
        Common values are .6 [1] or .75-.95 [2]. Can be given as array, but
        must then have a shape equal to (population.shape[0]//2,)

    Returns
    -------
    population : np.ndarray[bool]
        Population after crossover: operation performed in-place.
    """
    if not (((type(n_crossover) is int)
             and (0 < n_crossover < population.shape[1]))
            or (str(n_crossover).lower() == "uniform")):
        raise ValueError("n_crossover must be given a valid value")
    if n_crossover == "uniform":
        # Determine which individuals undergo crossover
        cross_bool = np.where(np.random.uniform(0, 1,
                                                (population.shape[0]//2,))
                              < p_crossover, True, False)
        # Perform uniform crossover
        # Loop through all sequential pairs
        for pair_idx in range(0, population.shape[0]//2, 2):
            if cross_bool[pair_idx]:
                # Determine which genes will undergo crossover
                crossover_mask = (np.random.uniform(0, 1,
                                                    population.shape[1]) < .5)
                # Store the crossover genes of the second individual
                temp_genes = np.copy(population[pair_idx+1, crossover_mask])
                # Set the crossed-over genes of the second individual
                population[pair_idx+1,
                           crossover_mask] = population[pair_idx,
                                                        crossover_mask]
                # Set the crossed-over genes of the first individual
                population[pair_idx, crossover_mask] = temp_genes
        return population
    else:
        # Perform n-point crossover; loop over all sequential pairs of
        # individuals
        # Determine which individuals undergo crossover
        cross_bool = np.where(np.random.uniform(0, 1,
                                                (population.shape[0]//2,))
                              < p_crossover, True, False)
        for pair_idx in range(0, population.shape[0]//2, 2):
            if cross_bool[pair_idx]:
                # First randomly determine the crossover points: label
                # the possible crossover points by the index of the gene
                # that they precede: the first crossover point has index 1,
                # the second 2, etc. up until the last, which has index
                # population.shape[1]-1 (as there is 1 less possible
                # crossover point than there are genes). This naming
                # allows us to later efficiently use slicing for cross-over.
                # Preallocate an array
                crossover_points = np.zeros((n_crossover + 2,),
                                            dtype=int)
                # All but the first and last components correspond to
                # crossover points
                crossover_points[1:-1] = np.sort(np.random.choice(
                    np.arange(1, population.shape[1]), n_crossover,
                    replace=False))
                # The first and last components are 0 and the gene length,
                # respectively; this allows us to loop neatly later on
                crossover_points[-1] = population.shape[1]
                # Loop over all crossover intervals, and exchange the
                for idx_start, idx_stop in zip(crossover_points[:-1],
                                               crossover_points[1:]):
                    # Store the crossover genes of the second individual
                    temp_genes = np.copy(
                        population[pair_idx+1, idx_start:idx_stop])
                    # Set the crossed-over genes for the second individual
                    population[pair_idx+1, idx_start:idx_stop] = \
                        population[pair_idx, idx_start:idx_stop]
                    # And for the first
                    population[pair_idx, idx_start:idx_stop] = temp_genes
        return population


def mating_selection(parents: np.ndarray[bool],
                     f_parents: np.ndarray[float],
                     offset: Union[float, bool] = None,
                     n_offspring: int = None,
                     select_best: int = 0) -> np.ndarray[bool]:
    """
    Select individuals in the population for reproduction.

    Individuals are selected for reproduction with either a probability given
    as:
    (1) p_selection[i] = (f_parents[i] - f_parents.min + offset)/
    (S_f - f_parents.min*mu + offset*mu),
    where S_f = sum(f_parents) and mu = f_parents.shape[0] (i.e. the number
    of parents). The offset denoted in this way must be positive.
    or (2) using ranked selection instead, with the reproduction probability
    scaling linearly with rank (and parents sorted according to increasing
    fitness). Which is used is determined by the offset parameter: if
    offset is set to None rather than a float value, ranked linear selection
    is used, otherwise roulette wheel selection is applied.

    Parameters
    ----------
    parents : np.ndarray[bool]
        Array containing the parent population, with each row comprising
        one individual.
    f_parents : np.ndarray[float]
        The function values of the parents, with each row containing the
        function value for the corresponding parent.
    offset : float or NoneType
        Offset to use for the proportional selection probability. Must be
        positive (note that the offset is defined differently than in the
        lecture slides; we automatically subtract the minimum function value).
        If set to None, use linear ranking selection instead.
    n_offspring : int or NoneType
        Number of offspring individuals to produce. If None, use the number of
        parents. Default is None.
    select_best : int
        Will select the first select_best populations for reproduction with
        certainty. Cannot be more than parents.shape[0]. Default is 0.

    Returns
    -------
    selected_population : np.ndarray[bool]
        Individuals in the population selected for reproduction.
    selected_f : np.ndarray[float]
        Their respective fitnesses.
    """
    # Denote the number of parents as mu
    mu = parents.shape[0]
    # Denote the bitstring length as ell
    ell = parents.shape[1]
    # Determine the number of offspring
    if n_offspring is None:
        n_offspring = mu
    # Determine the number to select probabilistically
    n_prob = n_offspring-select_best
    # Preallocate the offspring array
    selected_population = np.zeros((n_offspring, ell),
                                   dtype=bool)
    selected_f = np.zeros((n_offspring,), dtype=float)
    # Sort the parents according to fitness, in decreasing order
    argsort = np.argsort(f_parents)[::-1]
    if select_best > 0:
        # Sort the parents by performance; invert the result, as we want to
        # optimise rather than minimise
        # Select the select_best first parents for guaranteed reproduction
        for best_idx in range(select_best):
            # Select the corresponding parent
            selected_population[best_idx, :] = parents[argsort[best_idx], :]
            selected_f[best_idx] = f_parents[argsort[best_idx]]
    # Compute the probabilities of selection
    if offset is not None:
        f_min = f_parents.min()
        S_f = f_parents.sum()
        c = f_min - offset
        p_arr = (f_parents - c)/(S_f - c*mu + 1e-16)
    else:
        # Inefficient ad-hoc sorting algorithm that inserts the rank given
        # by argsort in the index of the array corresponding to each parent
        rank = np.zeros(f_parents.shape, dtype=int)
        for idx, arg in enumerate(argsort):
            rank[arg] = f_parents.shape[0] - idx
        rank_sum = np.sum(rank)
        p_arr = rank/rank_sum
    # Compute the cumulative distribution
    p_cum = np.cumsum(p_arr)
    # To avoid any potential float round-off issues, explicitly set the
    # final value to 1
    p_cum[-1] = 1
    # Spin the "roulette wheel"
    p_vals = np.random.uniform(size=(n_prob))
    for p_idx, p_val in enumerate(p_vals):
        # Find on which parent the roulette wheel landed
        idx = 0
        while (p_val > p_cum[idx]):
            idx += 1
        # Set this parent as the corresponding offspring individual
        selected_population[p_idx+select_best, :] = parents[idx, :]
        selected_f[p_idx+select_best] = f_parents[idx]
    return selected_population, selected_f

# %% Define auxiliary functions


def evaluate(problem, population: np.ndarray[bool]):
    """
    Evaluate the fitness value of the given population for the given problem.

    Evaluate the fitness values of the given population, as given by the
    selected problem.

    Parameters
    ----------
    problem : ioh.iohcpp.problem
        Problem to evaluate the fitness for.
    population : np.ndarray of bool
        Population, given as a 2D array of Booleans. The rows must correspond
        to the various individuals, while the columns must correspond to their
        genes. If an extended set of genes is used, this array must only
        contain those used directly in evaluation of the fitness (i.e. the
        genes must be directly passable to problem).

    Returns
    -------
    population_f : np.ndarray of float
        Fitness of the population, sorted in the rows corresponding to
        the individuals in population.
    """
    # Preallocate the fitness array
    population_f = np.zeros((population.shape[0],))
    # Loop over the population
    for ind_idx in range(population.shape[0]):
        population_f[ind_idx] = problem(np.uint8(population[ind_idx, :]))
    return population_f


# %%
# Final signature must be studentnumber1_studentnumber2_GA(problem)

def standard_GA(problem):
    """
    Optimise the given problem using a standard genetic algorithm.

    Parameters
    ----------
    problem : ioh.iohcpp.problem
        Problem in question.

    Returns
    -------
    None.
    """
    # Set the required parameters
    pop_size = 50  # population size
    offset = 0.  # Offset used in population selection; defined according to
    # the argument offset of mating_selection()
    n_crossover = "uniform"  # Number of crossover points
    p_crossover = 0.9  # Probability of crossover between a pair of parents
    p_mutation = np.array([1/dimension])  # Gene-wise mutation probability
    select_best = 1  # Number of top-performers to keep per selection round
    # randomly initiate the first population
    pop = np.random.uniform(0, 1, (pop_size, dimension)) > .5
    # Evaluate the fitnesses of this population
    f_pop = evaluate(problem, pop)
    # `problem.state.evaluations` counts the number of function evaluation
    # automatically, which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.
    while problem.state.evaluations < budget:
        # please implement the mutation, crossover, selection here
        # First select the parents
        pop, _ = mating_selection(pop, f_pop, offset=offset,
                                  select_best=select_best)
        # Perform crossover
        pop = crossover(pop, n_crossover=n_crossover,
                        p_crossover=p_crossover)
        # Perform mutation
        pop = mutate(pop, p_mutation=p_mutation)
        # Evaluate the new generation
        f_pop = evaluate(problem, pop)

        # this is how you evaluate one solution `x`
        # f = problem(x)
        # no return value needed


def self_adjust_GA(problem):
    """
    Optimise the given problem using a self-adjusting genetic algorithm.

    Uses a self-adjusting algorithm like that in [1], which relies only on the
    parameter F that dictates the speed at which the mutation probability is
    altered.

    [1] Doerr, B., Gießen, C., Witt, C., & Yang, J. (2019). The (1 + λ)
    Evolutionary Algorithm with Self-Adjusting Mutation Rate. Algorithmica,
    81(2), 593–631. https://doi.org/10.1007/s00453-018-0502-x

    Parameters
    ----------
    problem : ioh.iohcpp.problem
        Problem in question.

    Returns
    -------
    None.
    """
    # Set the required parameters
    pop_size = 3  # population size
    offset = 0.  # Offset used in population selection; defined according to
    # the argument offset of mating_selection()
    n_crossover = "uniform"  # Number of crossover points
    p_crossover = 0.5  # Probability of crossover between two consecutive pairs
    # of parents
    p_mutation = np.array([1/dimension])  # Initial mutation probability
    # from [1]
    F = 1.1  # Mutation rate change factor
    select_best = 0  # Number of top-performers to keep per selection round

    # randomly initiate the first population
    pop = np.random.uniform(0, 1, (pop_size, dimension)) > .5
    # Evaluate the fitnesses of this population
    f_pop = evaluate(problem, pop)
    # `problem.state.evaluations` counts the number of function evaluation
    # automatically, which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.
    while problem.state.evaluations < budget:
        # First select the parents
        pop, _ = mating_selection(pop, f_pop, offset=offset,
                                  select_best=select_best)
        # Perform crossover
        pop = crossover(pop, n_crossover=n_crossover,
                        p_crossover=p_crossover)
        # Perform mutation
        # Divide the groups into two sub-groups;
        group_idx = int(np.ceil(pop.shape[0]/2))
        # For group 1 (with higher mutation)
        pop1 = mutate(pop[:group_idx, :], p_mutation=F*p_mutation)
        # For group 2 (with lower mutation)
        pop2 = mutate(pop[group_idx:, :], p_mutation=p_mutation/F)
        # Evaluate the new generation
        f_pop1 = evaluate(problem, pop1)
        f_pop2 = evaluate(problem, pop2)
        # Adjust the mutation rate
        if np.random.uniform(0, 1, 1) > .5:
            # With 50% chance, set the mutation rate to that which performed
            # best in this generation
            if np.max(f_pop1 >= f_pop2):
                p_mutation = F*p_mutation
            else:
                p_mutation = p_mutation/F
        else:
            # With the other 50% probability, perturb it randomly
            # but clip the result to be within the range [2/dimension, 1/4],
            # as outside those ranges we get a subpopulation with nonsensical
            # mutation rates in the next generation
            p_mutation = np.clip(np.random.uniform(1/2, 2)*p_mutation,
                                 2/dimension,
                                 1/4)
        # Store the resulting population and their function values
        pop[:group_idx, :] = pop1
        pop[group_idx:, :] = pop2
        f_pop[:group_idx] = f_pop1
        f_pop[group_idx:] = f_pop2


def self_adjust_GA2(problem):
    """
    Optimise the given problem using a self-adjusting genetic algorithm.

    Uses a self-adjusting algorithm like that in [1].

    [1] Srinivas, M., & Patnaik, L. M. (1994). Adaptive Probabilities of
    Crossover Genetic in Mutation and Algorithms. IEEE TRANSACTIONS ON
    SYSTEMS, MAN AND CYBERNETICS, 24(4), 656–667.

    Parameters
    ----------
    problem : ioh.iohcpp.problem
        Problem in question.

    Returns
    -------
    None.
    """
    # Set the required parameters
    pop_size = 60  # population size
    offset = 0.  # Offset used in population selection; defined according to
    # the argument offset of mating_selection()
    n_crossover = "uniform"  # Number of crossover points
    select_best = 0  # Number of top-performers to keep per selection round

    # Set the tuning parameters
    k1 = 1.  # Crossover scaling parameter
    k2 = .5  # Mutation scaling parameter
    k3 = 1.  # Crossover upper bound
    k4 = .5  # Mutation upper bound

    # randomly initiate the first population
    pop = np.random.uniform(0, 1, (pop_size, dimension)) > .5
    # Evaluate the fitnesses of this population
    f_pop = evaluate(problem, pop)
    # `problem.state.evaluations` counts the number of function evaluation
    # automatically, which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.
    while problem.state.evaluations < budget:
        # First select the parents
        pop, f_pop = mating_selection(pop, f_pop, offset=offset,
                                      select_best=select_best)
        # Determine the crossover and mutation probabilities
        # Compute the auxiliary quantities we need
        f_max = np.max(f_pop)
        variation = f_max - np.mean(f_pop) + 1e-32
        # Determine the mean fitness between all parent pairs
        f_mean = np.zeros(f_pop.shape[0]//2,)
        for f_mean_idx, idx in enumerate(range(0, f_pop.shape[0]-1, 2)):
            f_mean[f_mean_idx] = (f_pop[idx] + f_pop[idx+1])/2
        # Calculate the resulting crossover probability per pair
        pc_arr = np.clip(k1*(f_max - f_mean)/variation, 0, k3)
        # Calculate the mutation probability
        pm_arr = np.clip(k2*(f_max - f_pop)/variation, 1/dimension, k4)

        # Perform crossover
        pop = crossover(pop, n_crossover=n_crossover,
                        p_crossover=pc_arr)
        # Perform mutation
        pop = mutate(pop, pm_arr)
        # Evaluate the new fitnesses
        f_pop = evaluate(problem, pop)


def two_rate_GA(problem):
    """
    Optimise the given problem using a two-rate genetic algorithm.

    Uses a two-rate (1 + λ)-GA like that in [1].

    [1] Doerr, B., Gießen, C., Witt, C., & Yang, J. (2019). The (1 + λ)
    Evolutionary Algorithm with Self-Adjusting Mutation Rate. Algorithmica,
    81(2), 593–631. https://doi.org/10.1007/s00453-018-0502-x

    Parameters
    ----------
    problem : ioh.iohcpp.problem
        Problem in question.

    Returns
    -------
    None.
    """
    r_init = 1
    F = 2
    # Randomly create the initial "population" of one string
    x = np.uint8(np.random.uniform(0, 1, size=(dimension,)) < .5)
    f_x = problem(x)
    # Set the initial rate
    r = r_init
    # Set the nr. of offspring
    lam = 25
    while problem.state.evaluations < budget:
        # Initialise the array of function-values
        offspring_f = np.zeros((lam,))
        # and of offspring themselves
        offspring = np.zeros((lam, dimension), dtype=np.uint8)
        # Copy the parent into the offspring array
        offspring[:, :] = x[None, :]
        for i in range(10):
            if i <= lam/2:
                p_mutate = r/(F*dimension)
            else:
                p_mutate = F*r/dimension
            # Create a mask that determines which bits are mutated
            mut_mask = (np.random.uniform(0, 1, x.shape) < p_mutate)
            offspring[i, mut_mask] = 1 - offspring[i, mut_mask]
            offspring_f[i] = problem(offspring[i, :])
        x_star = np.argmax(offspring_f)
        if offspring_f[x_star] >= f_x:
            x = np.copy(offspring[x_star, :])
            f_x = np.copy(offspring_f[x_star])
        if np.random.uniform(0, 1) < .5:
            if x_star <= lam/2:
                r = r/(F*dimension)
            else:
                r = F*r/dimension
        else:
            r = np.random.choice([r/(F*dimension), F*r/dimension])
        r = np.clip(r, 2, dimension/4)


def two_rate_mu_GA(problem):
    """
    Optimise the given problem using a two-rate genetic algorithm.

    Uses a two-rate (μ + λ)-GA, which generalises that in [1].

    [1] Doerr, B., Gießen, C., Witt, C., & Yang, J. (2019). The (1 + λ)
    Evolutionary Algorithm with Self-Adjusting Mutation Rate. Algorithmica,
    81(2), 593–631. https://doi.org/10.1007/s00453-018-0502-x

    Parameters
    ----------
    problem : ioh.iohcpp.problem
        Problem in question.

    Returns
    -------
    None.
    """
    # Set the parameters of the GA
    mu = 5
    lam = 20
    r_init = 1
    F = 1.1
    # Randomly create the initial population
    parents = np.uint8(np.random.uniform(0, 1, size=(mu, dimension)) < .5)
    f_parents = evaluate(problem, parents)
    # Set the initial rate
    r = r_init
    while problem.state.evaluations < budget:
        # Initialise the array of function-values
        offspring_f = np.zeros((lam,))
        # and of offspring themselves
        offspring, _ = mating_selection(parents, f_parents,
                                        offset=0., n_offspring=lam)
        # Mutate the offspring
        # Create a mask that determines which bits are mutated
        mut_mask = np.zeros((lam, dimension), dtype=bool)
        # First group (lower mutation)
        mut_mask[:lam//2, :] = (
            np.random.uniform(0, 1,
                              (lam//2, parents.shape[1])) <
            r/(F*dimension))
        # Second group (higher mutation)
        mut_mask[lam//2:, :] = (
            np.random.uniform(0, 1,
                              (lam - lam//2, parents.shape[1])) <
            F*r/dimension)
        offspring[mut_mask] = 1 - offspring[mut_mask]
        offspring_f = evaluate(problem, offspring)
        # Select the best mu offspring + parents to provide the next gen
        arg_nextgen = np.argsort(offspring_f)[::-1][:mu]
        parents = offspring[arg_nextgen, :]
        f_parents = offspring_f[arg_nextgen]
        # Determine the mutation rate for the next generation:
        # here we have the best-performing individual play the role that the
        # single parent would in the single-parent two-rate GA
        if np.random.uniform(0, 1) < .5:
            if arg_nextgen[0] <= lam/2:
                r = r/(F*dimension)
            else:
                r = F*r/dimension
        else:
            r = np.random.choice([r/(F*dimension), F*r/dimension])
        r = np.clip(r, 2, dimension/4)



def create_problem(fid: int):
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension,
                          instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output.
    # You should compress the folder 'run' and upload it to IOHanalyzer.
    time_str = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    log = logger.Analyzer(
        # the working directory in which a folder named `folder_name`
        # (the next argument) will be created to store data
        root="data",
        folder_name="run_"+time_str,  # the folder name to which the raw
        # performance
        # data will be stored
        algorithm_name="GA_"+time_str,  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(log)
    return problem, log


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    F18, _logger = create_problem(18)
    for run in range(20):
        two_rate_mu_GA(F18)
        F18.reset()  # it is necessary to reset the problem after
        # each independent run
    _logger.close()  # after all runs, it is necessary to close the
    # logger to make sure all data are written to the folder

    F19, _logger = create_problem(19)
    for run in range(20):
        two_rate_mu_GA(F19)
        F19.reset()
    _logger.close()

    F1, _logger = create_problem(1)
    for run in range(20):
        two_rate_mu_GA(F1)
        F1.reset()
    _logger.close()
