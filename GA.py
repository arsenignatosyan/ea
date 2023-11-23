import numpy as np
# you need to install this package `ioh`. Please see documentations here:
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
from ioh import get_problem, logger, ProblemClass

budget = 5000
dimension = 50

# To make your results reproducible (not required by the assignment),
# you could set the random seed by
np.random.seed(3366766)

# %% Define the variators and selection functions


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
