import numpy as np
# you need to install this package `ioh`. Please see documentations here:
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass
from typing import Union, Optional

budget = 5000
dimension = 50

# To make your results reproducible (not required by the assignment),
# you could set the random seed by
np.random.seed(3366766)

# %% Define the variator and selection functions


def mutate(population: np.ndarray[bool],
           p_mutation: Union[float, None] = None):
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
    p_mutation : float or NoneType, optional
        Probability of mutation per gene. If set to None, the "standard" value
        of 1/population.shape[1] is used, which flips on average one gene
        per individual. The default is None.

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
    mutate_array = (np.random.uniform(0, 1, population.shape) < p_mutation)
    # Flip the selected bits in-place using array masking
    # We use the fact that Python uses 0 and 1 interchangeably with False and
    # True to do this quickly.
    population[mutate_array] = 1 - population[mutate_array]
    return population


def crossover(population: np.ndarray[bool],
              n_crossover: Union[int, str],
              p_crossover: float,
              p_gene_crossover: Union[float, None] = None):
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
    p_crossover : float
        Probability that a given sequential pair undergoes crossover.
        Common values are .6 [1] or .75-.95 [2].
    p_gene_crossover : float or NoneType, optional
        If n_crossover is set to "uniform", this determines the probability
        of crossover per individual gene. If n_crossover is set to an integer
        value, this parameter is not used and can be set to None.

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
        if p_gene_crossover is None:
            raise ValueError("Set a valid value for p_gene_crossover")
        # Perform uniform crossover
        # Loop through all sequential pairs, starting with the pair comprising
        # the -1-th and 0-th individual.
        for pair_idx in range(-1, population.shape[0]-1):
            if np.random.uniform(0, 1) < p_crossover:
                # Determine which genes will undergo crossover
                crossover_mask = (np.random.uniform(0, 1,
                                                    population.shape[1]) <
                                  p_gene_crossover)
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
        for pair_idx in range(-1, population.shape[0]-1):
            if np.random.uniform(0, 1) < p_crossover:
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
                     offset: float,
                     n_offspring: int = None) -> np.ndarray[bool]:
    """
    Select individuals in the population for reproduction.

    Individuals are selected for reproduction with a probability given as
    p_selection[i] = (f_parents[i] - f_parents.min + offset)/
    (S_f - f_parents.min*mu + offset*mu),
    where S_f = sum(f_parents) and mu = f_parents.shape[0] (i.e. the number
    of parents). The offset denoted in this way must be positive.

    Parameters
    ----------
    parents : np.ndarray[bool]
        Array containing the parent population, with each row comprising
        one individual.
    f_parents : np.ndarray[float]
        The function values of the parents, with each row containing the
        function value for the corresponding parent.
    offset : float
        Offset to use for the proportional selection probability. Must be
        positive (note that the offset is defined differently than in the
        lecture slides; we automatically subtract the minimum function value).
    n_offspring : int or NoneType
        Number of offspring individuals to produce. If None, use the number of
        parents. Default is None.

    Returns
    -------
    selected_population : np.ndarray[bool]
        Individuals in the population selected for reproduction.
    """
    # Denote the number of parents as mu
    mu = parents.shape[0]
    # Denote the bitstring length as ell
    ell = parents.shape[1]
    # Determine the number of offspring
    if n_offspring is None:
        n_offspring = mu
    # Preallocate the offspring array
    selected_population = np.zeros((n_offspring, ell),
                                   dtype=bool)
    # Compute the probabilities of selection
    f_min = f_parents.min()
    S_f = f_parents.sum()
    c = f_min - offset
    p_arr = (f_parents - c)/(S_f - c*mu)
    # Compute the cumulative distribution
    p_cum = np.cumsum(p_arr)
    # To avoid any potential float round-off issues, explicitly set the
    # final value to 1
    p_cum[-1] = 1
    # Spin the "roulette wheel"
    p_vals = np.random.uniform(size=(n_offspring))
    for p_idx, p_val in enumerate(p_vals):
        # Find on which parent the roulette wheel landed
        gen_idx = np.nonzero(p_cum <= p_val)[0]
        if gen_idx.size == 0:
            gen_idx = 0
        else:
            gen_idx = gen_idx[-1]
        # Set this parent as the corresponding offspring individual
        selected_population[p_idx, :] = parents[gen_idx, :]
    return selected_population

    # %%


def studentnumber1_studentnumber2_GA(problem):
    # initial_pop = ... make sure you randomly create the first population

    # `problem.state.evaluations` counts the number of function evaluation
    # automatically, which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.
    while problem.state.evaluations < budget:
        # please implement the mutation, crossover, selection here
        # .....
        # this is how you evaluate one solution `x`
        # f = problem(x)
        # no return value needed


def create_problem(fid: int):
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension,
                          instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output.
    # You should compress the folder 'run' and upload it to IOHanalyzer.
    log = logger.Analyzer(
        # the working directory in which a folder named `folder_name`
        # (the next argument) will be created to store data
        root="data",
        folder_name="run",  # the folder name to which the raw performance
        # data will be stored
        algorithm_name="genetic_algorithm",  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(log)
    return problem, log


if __name__ == "__main__":
    # this how you run your algorithm with 20 repetitions/independent run
    F18, _logger = create_problem(18)
    for run in range(20):
        studentnumber1_studentnumber2_GA(F18)
        F18.reset()  # it is necessary to reset the problem after
        # each independent run
    _logger.close()  # after all runs, it is necessary to close the
    # logger to make sure all data are written to the folder

    F19, _logger = create_problem(19)
    for run in range(20):
        studentnumber1_studentnumber2_GA(F19)
        F19.reset()
    _logger.close()
