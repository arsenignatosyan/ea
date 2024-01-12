from datetime import datetime
from typing import Tuple, Union, List, Literal

import ioh
import numpy as np
from ioh import get_problem, logger, ProblemClass

# globally defined variables that hold for all ES settings
np.random.seed(42)
budget = 5000
dimension = 50
bit_chunk = 5


def initialize_population(problem: Union[ioh.iohcpp.problem.IsingRing, ioh.iohcpp.problem.LABS],
                          initial_population_size: int) -> Tuple[np.ndarray[bool], List[float]]:
    """
    Generates initial population for the optimization of the problem, also evaluates the generated population.

    Parameters
    ----------
    problem: Union[ioh.iohcpp.problem.IsingRing, ioh.iohcpp.problem.LABS]
        Problem to optimize for.
    initial_population_size: int
        Number of individuals to generate.
    mutation_type: Literal["single", "individual", "correlated"]
        The mutation_type of the ES based on which the angles will be generated or not.

    Returns
    -------
    population: np.ndarray[bool]
        The generated population of bit vectors.
    population_f: np.ndarray[float]
        The evaluation of the generated population.
    """
    population = np.random.randint(2, size=(initial_population_size, problem.meta_data.n_variables))
    population_f = [problem(population_item) for population_item in population]

    return population, population_f


def generate_initial_population_sigma(population: np.ndarray[float], initial_step_size: float,
                                      mutation_type: Literal["single", "individual", "correlated"]) -> \
        np.ndarray[float]:
    """
    Generates initial step sizes for the population given the initial_step_size and the mutation_type.

    Parameters
    ----------
    population: np.ndarray[float]
        The population for which the step sizes need to be generated.
    initial_step_size: float
        The value of the step size to be generated.
    mutation_type: Literal["single", "individual", "correlated"]
        The mutation_type of the ES based on which the angles will be generated or not.

    Returns
    -------
    out: np.ndarray[float]
        The generated step sizes.
    """
    if mutation_type == "single":
        population_sigma = np.ones(population.shape[0])
    else:
        population_sigma = np.ones(population.shape)

    return population_sigma * initial_step_size


def generate_rotation_angles(mutation_type: Literal["single", "individual", "correlated"], defined_population_size: int,
                             problem_dimension: int) -> Union[np.ndarray[float], None]:
    """
    Generates rotational angles from normal distribution of mean 0 and std pi, if the mutation_type is "correlated".
    Else, no rotation angles need to be generated.

    Parameters
    ----------
    mutation_type: Literal["single", "individual", "correlated"]
        The mutation_type of the ES based on which the angles will be generated or not.
    defined_population_size: int
        The defined size of initial population.
    problem_dimension: int
        The dimension of the problem.

    Returns
    -------
    out: Union[np.ndarray[float], None]
        The generated rotation angles.
    """
    rotation_angles = None
    if mutation_type == "correlated":
        k = int(problem_dimension * (problem_dimension - 1) / 2)
        rotation_angles = np.random.normal(0, np.pi, size=(defined_population_size, k))

    return rotation_angles


def one_sigma_mutation(population: np.ndarray[float], population_sigma: np.ndarray[float], problem_dimension: int) -> \
        Tuple[np.ndarray[float], np.ndarray[float]]:
    """
    Mutates the given population and its strategy parameters using the one sigma mutation.

    Parameters
    ----------
    population: np.ndarray[float]
        The encoded population of float values to be mutated.
    population_sigma: np.ndarray[float]
        The step sizes of population to be mutated.
    problem_dimension: int
        The dimension of the problem.

    Returns
    -------
    mut_population: np.ndarray[float]
        The mutated individuals.
    mut_sigma: np.ndarray[float]
        The step sizes of the mutated individuals.
    """
    tau = 1 / np.sqrt(problem_dimension)
    mut_sigma = [s * np.exp(tau * np.random.normal()) for s in population_sigma]
    mut_population = [p + s * np.random.normal() for p, s in zip(population, mut_sigma)]
    return np.array(mut_population), np.array(mut_sigma)


def individual_sigma_mutation(population: np.ndarray[float], population_sigma: np.ndarray[float],
                              problem_dimension: int) -> Tuple[np.ndarray[float], np.ndarray[float]]:
    """
    Mutates the given population and its strategy parameters using the individual sigma mutation.

    Parameters
    ----------
    population: np.ndarray[float]
        The encoded population of float values to be mutated.
    population_sigma: np.ndarray[float]
        The step sizes of population to be mutated.
    problem_dimension: int
        The dimension of the problem.

    Returns
    -------
    mut_population: np.ndarray[float]
        The mutated individuals.
    mut_sigma: np.ndarray[float]
        The step sizes of the mutated individuals.
    """
    tau = 1 / np.sqrt(2 * problem_dimension)
    tau_prime = 1 / np.sqrt(2 * np.sqrt(problem_dimension))
    g = np.random.normal()
    mut_population = np.empty(population.shape)
    mut_sigma = np.empty(population_sigma.shape)
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            mut_sigma[i][j] = population_sigma[i][j] * np.exp(g * tau_prime + tau * np.random.normal())
            mut_population[i][j] = population_sigma[i][j] + mut_sigma[i][j] * np.random.normal()

    return mut_population, mut_sigma


def rotation_matrix(alpha_ij: float, i: int, j: int, n: int) -> np.ndarray[float]:
    """
    Constructs the rotation matrix based on the provided angle alpha_ij, and indices i and j.

    Parameters
    ----------
    alpha_ij: float
        The rotation angle between elements i and j.
    i: int
        The i-th element.
    j: int
        The j-th element.
    n: int
        The dimension of the problem.

    Returns
    -------
    out: np.ndarray[float]
        The constructed rotation matrix.
    """
    R = np.eye(n)
    cos_alpha, sin_alpha = np.cos(alpha_ij), np.sin(alpha_ij)

    R[i, i] = cos_alpha
    R[j, j] = cos_alpha
    R[i, j] = -sin_alpha
    R[j, i] = sin_alpha

    return R


def multiply_rotation_matrix(alpha: np.ndarray[float], n: int) -> np.ndarray[float]:
    """
    Calculates the rotation matrices based on the passed angles and multiplies them.

    Parameters
    ----------
    alpha: np.ndarray[float]
        The passed rotation angles.
    n: int
        The dimension of the problem.

    Returns
    -------
    out: np.ndarray[float]
        The multiplied matrix of the rotation matrices.
    """
    mult_matrix = np.eye(n)
    count = 0
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            alpha_ij = alpha[count]
            mult_matrix = np.matmul(mult_matrix, rotation_matrix(alpha_ij=alpha_ij, i=i, j=j, n=n))
            count += 1

    return mult_matrix


def correlated_mutation(population: np.ndarray[float], population_sigma: np.ndarray[float],
                        population_rotation_angles: np.ndarray[float],
                        problem_dimension: int) -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """
    Mutates the given population and its strategy parameters using the correlated mutation.

    Parameters
    ----------
    population: np.ndarray[float]
        The encoded population of float values to be mutated.
    population_sigma: np.ndarray[float]
        The step sizes of population to be mutated.
    population_rotation_angles: Union[np.ndarray[float], None]
        The rotation angles of population to be mutated.
    problem_dimension: int
        The dimension of the problem.

    Returns
    -------
    mut_population: np.ndarray[float]
        The mutated individuals.
    mut_sigma: np.ndarray[float]
        The step sizes of the mutated individuals.
    mut_rotation_angles: Union[np.ndarray[float], None]
        The rotation angles of the mutated individuals.
    """
    tau = 1 / np.sqrt(2 * problem_dimension)
    tau_prime = 1 / np.sqrt(2 * np.sqrt(problem_dimension))
    g = np.random.normal()
    beta = np.pi / 36
    mut_population = np.empty(population.shape)
    mut_sigma = np.empty(population_sigma.shape)
    mut_rotation_angles = np.empty(population_rotation_angles.shape)
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            mut_sigma[i][j] = population_sigma[i][j] * np.exp(g * tau_prime + tau * np.random.normal())

        for k in range(population_rotation_angles.shape[1]):
            mut_rotation_angles[i][k] = population_rotation_angles[i][k] + np.random.normal(0, beta)
            if abs(mut_rotation_angles[i][k]) > np.pi:
                mut_rotation_angles[i][k] = mut_rotation_angles[i][k] - 2 * np.pi * np.sign(mut_rotation_angles[i][k])

        sigma_mat = np.diag(mut_sigma[i])
        mult_rotation_matrix = multiply_rotation_matrix(alpha=mut_rotation_angles[i], n=problem_dimension)
        C_sqrt = np.matmul(mult_rotation_matrix, sigma_mat)
        C_prime = np.matmul(C_sqrt, C_sqrt.T)
        mut_population[i] = population[i] + mut_sigma[i] * np.random.multivariate_normal(
            mean=np.zeros(problem_dimension), cov=C_prime)

    return mut_population, mut_sigma, mut_rotation_angles


def mutation(mutation_type: Literal["single", "individual", "correlated"], population: np.ndarray[float],
             population_sigma: np.ndarray[float], population_rotation_angles: np.ndarray[float],
             problem_dimension: int) -> Tuple[np.ndarray[float], np.ndarray[float], Union[np.ndarray[float], None]]:
    """
    Mutates the given population and its strategy parameters using the provided mutation_type.

    Parameters
    ----------
    mutation_type: Literal["single", "individual", "correlated"]
        The type of mutation to use for mutating individuals.
    population: np.ndarray[float]
        The encoded population of float values to be mutated.
    population_sigma: np.ndarray[float]
        The step sizes of population to be mutated.
    population_rotation_angles: Union[np.ndarray[float], None]
        The rotation angles of population to be mutated, if the mutation_type passed is "correlated".
    problem_dimension: int
        The dimension of the problem.

    Returns
    -------
    mut_population: np.ndarray[float]
        The mutated individuals.
    mut_sigma: np.ndarray[float]
        The step sizes of the mutated individuals.
    mut_rotation_angles: Union[np.ndarray[float], None]
        The rotation angles of the mutated individuals.
    """
    mut_rotation_angles = None
    if mutation_type == "correlated":
        mut_population, mut_sigma, mut_rotation_angles = correlated_mutation(
            population=population,
            population_sigma=population_sigma,
            population_rotation_angles=population_rotation_angles,
            problem_dimension=problem_dimension
        )
    elif mutation_type == "individual":
        mut_population, mut_sigma = individual_sigma_mutation(population=population, population_sigma=population_sigma,
                                                              problem_dimension=problem_dimension)
    else:
        mut_population, mut_sigma = one_sigma_mutation(population=population, population_sigma=population_sigma,
                                                       problem_dimension=problem_dimension)

    return mut_population, mut_sigma, mut_rotation_angles


def discrete_recombination(parent: np.ndarray[bool]) -> np.ndarray[bool]:
    """
    The discrete recombination takes two random vector from the parent each element of the offspring is chosen randomly
    from parent1 or parent2.

    Parameters
    ----------
    parent: np.ndarray[float]
        The parent parameters.

    Returns
    -------
    out: np.ndarray[float]
        The offspring parameters.
    """
    sample_idxs = np.random.choice(len(parent), 2)
    sample_parent = parent[sample_idxs]
    offspring = [sample_parent[np.random.randint(len(sample_parent))][idx] for idx in range(parent.shape[1])]

    return np.array(offspring)


def global_intermediate_recombination(parent: np.ndarray[float]) -> Tuple[np.ndarray[float]]:
    """
    The global intermediate recombination takes the average of all parents.

    Parameters
    ----------
    parent: np.ndarray[float]
        The parent parameters.

    Returns
    -------
    out: np.ndarray[float]
        The offspring parameters.
    """
    offspring = np.mean(parent, axis=0)

    return offspring


def preserve_recombination(parent: np.ndarray[float]) -> np.ndarray[float]:
    """
    The preserve recombination takes random vector from the parent and assigns it to an offspring.

    Parameters
    ----------
    parent: np.ndarray[float]
        The parent parameters.

    Returns
    -------
    out: np.ndarray[float]
        The offspring parameters.
    """
    parent_idx = np.random.choice(len(parent))
    offspring = parent[parent_idx]

    return offspring


def recombination(lambda_: int, binary_population: np.ndarray[bool], population_sigma: np.ndarray[float],
                  population_rotation_angles: Union[np.ndarray[float], None]) -> Tuple[
    np.ndarray[bool], np.ndarray[float], np.ndarray[float]]:
    """
    Using recombination technique create lambda_ new offsprings from the given population and its strategy parameters.
    For x-s the discrete recombination is chosen. For step sizes the global intermediate recombination is selected.
    For rotation angles the preserving recombination is used.

    Parameters
    ----------
    lambda_: int
        The number of offsprings to get from using recombination.
    binary_population: np.ndarray[bool]
        The population in its original form.
    population_sigma: np.ndarray[float]
        The step sizes of population.
    population_rotation_angles: Union[np.ndarray[float], None]
        The rotation angles of population, if the mutation_type of the ES is "correlated".

    Returns
    -------
    offspring: np.ndarray[bool]
        The selected individuals in its original form.
    offspring_sigma: np.ndarray[float]
        The step sizes of the selected individuals.
    offspring_rotation_angles: np.ndarray[float]
        The rotation angles of the selected individuals, if the mutation_type of the ES is "correlated".
    """
    offspring, offspring_sigma, offspring_rotation_angles = [], [], []
    for _ in range(lambda_):
        o = discrete_recombination(parent=binary_population)
        offspring.append(o)
        o_sigma = global_intermediate_recombination(parent=population_sigma)
        offspring_sigma.append(o_sigma)
        if (population_rotation_angles is not None) and len(population_rotation_angles):
            o_rotation_angles = preserve_recombination(parent=population_rotation_angles)
            offspring_rotation_angles.append(o_rotation_angles)

    return np.array(offspring), np.array(offspring_sigma), np.array(offspring_rotation_angles)


def encode_to_real(binary_population: np.ndarray[bool]) -> np.ndarray[int]:
    """
    Encodes the passed population of bitstring into vectors of real values, by dividing the bitsring into chunks and
    converting the chunks into integers.

    Parameters
    ----------
    binary_population: np.ndarray[bool]
        The population in its original form that needs to be encoded into real vectors.

    Returns
    -------
    out: np.ndarray[int]
        The encoded individual in the form of int vectors.
    """
    real_population = []
    for bit_vector in binary_population:
        bit_vector = [str(b) for b in bit_vector]
        bit_vector_str = ''.join(bit_vector)
        real_vector = [int(bit_vector_str[i:i + bit_chunk], 2) for i in range(0, len(bit_vector), bit_chunk)]
        real_population.append(real_vector)

    return np.array(real_population)


def decode_to_binary(real_population: np.ndarray[float]) -> np.ndarray[bool]:
    """
    Decoded the passed population, a vector of real values into a vector of binary values

    Parameters
    ----------
    real_population: np.ndarray[float]
        The population of float values that need to be decoded to binary.

    Returns
    -------
    out: np.ndarray[bool]
        The decoded individuals in the form of bit vectors.
    """
    real_population = real_population.round().astype(int)

    binary_population = []
    for i, real_vector in enumerate(real_population):
        for j in range(len(real_vector)):
            value_j = real_vector[j]
            if value_j >= pow(2, bit_chunk):
                real_vector[j] = 31
            elif value_j < 0:
                real_vector[j] = 0
        bit_vector = ['{0:05b}'.format(r) for r in real_vector]
        bit_vector = [[*b] for b in bit_vector]
        bit_vector_flat = [int(item) for sublist in bit_vector for item in sublist]
        binary_population.append(bit_vector_flat)

    return np.array(binary_population)


def selection(mu_: int, selection_type: Literal[",", "+"], real_population: np.ndarray[float],
              binary_population: np.ndarray[bool], real_offspring: np.ndarray[float],
              binary_offspring: np.ndarray[bool], population_sigma: np.ndarray[float],
              offspring_sigma: np.ndarray[float], population_f: np.ndarray[float], offspring_f: np.ndarray[float],
              population_rotation_angles: Union[np.ndarray[float], None],
              offspring_rotation_angles: Union[np.ndarray[float], None]) -> Tuple[
    np.ndarray[float], np.ndarray[bool], np.ndarray[float], Union[np.ndarray[float], None], np.ndarray[float]]:
    """
    Select mu_ offspring based on the selection type chosen

    Parameters
    ----------
    mu_: int
        The number of individual to select.
    selection_type: Literal[",", "+"]
        The type of selection to use for selecting new individuals.
    real_population: np.ndarray[float]
        The encoded population of float values that may be considered for selection if the type is "+".
    binary_population: np.ndarray[bool]
        The population in its original form that may be considered for selection if the type is "+".
    real_offspring: np.ndarray[float]
        The encoded offsprings of float values.
    binary_offspring: np.ndarray[bool]
        The offsprings in its original form.
    population_sigma: np.ndarray[float]
        The step sizes of population.
    offspring_sigma: np.ndarray[float]
        The step sizes of offsprings.
    population_f: np.ndarray[float]
        Values of population evaluation.
    offspring_f: np.ndarray[float]
        Values of offsprings evaluation.
    population_rotation_angles: Union[np.ndarray[float], None]
        The rotation angles of population, if the mutation_type of the ES is "correlated".
    offspring_rotation_angles: Union[np.ndarray[float], None]
        The rotation angles of offsprings, if the mutation_type of the ES is "correlated".

    Returns
    -------
    new_real: np.ndarray[float]
        The encoded selected individuals of float values.
    new_binary: np.ndarray[bool]
        The selected individuals in its original form.
    new_sigma: np.ndarray[float]
        The step sizes of the selected individuals.
    new_rotation_angles: Union[np.ndarray[float], None]
        The rotation angles of the selected individuals, if the mutation_type of the ES is "correlated".
    new_f: np.ndarray[float]
        Values of the selected individuals evaluation.
    """
    if selection_type == ",":
        considered_real = real_offspring
        considered_binary = binary_offspring
        considered_sigma = offspring_sigma
        considered_rotation_angles = offspring_rotation_angles
        considered_f = offspring_f
    else:
        considered_real = np.append(arr=real_population, values=real_offspring, axis=0)
        considered_binary = np.append(arr=binary_population, values=binary_offspring, axis=0)
        considered_sigma = np.append(arr=population_sigma, values=offspring_sigma, axis=0)
        if population_rotation_angles is not None:
            considered_rotation_angles = np.append(arr=population_rotation_angles, values=offspring_rotation_angles,
                                                   axis=0)
        else:
            considered_rotation_angles = None
        considered_f = np.append(arr=population_f, values=offspring_f, axis=0)
    min_idx = np.argsort(considered_f)[-mu_:]
    new_real = considered_real[min_idx]
    new_binary = considered_binary[min_idx]
    new_sigma = considered_sigma[min_idx]
    new_rotation_angles = considered_rotation_angles[min_idx] if considered_rotation_angles is not None else None
    new_f = considered_f[min_idx]

    return new_real, new_binary, new_sigma, new_rotation_angles, new_f


def s4034538_s3366766_ES(problem: Union[ioh.iohcpp.problem.IsingRing, ioh.iohcpp.problem.LABS], mu_: int, lambda_: int,
                         initial_step_size: float, mutation_type: Literal["single", "individual", "correlated"],
                         selection_type: Literal[",", "+"]):
    """
    Optimizes for the given problem using Evolution Strategy (ES)

    Parameters
    ----------
    problem: Union[ioh.iohcpp.problem.IsingRing, ioh.iohcpp.problem.LABS]
        Problem to optimize for.
    mu_: int
        Number of parents to select for reproduction in each generation.
    lambda_: int
        Number of offspring to produce from the parents in each generation.
    initial_step_size: float
        The step size that should be initially set for each individual in the population.
    mutation_type: Literal["single", "individual", "correlated"]
        The type of mutation to use for mutating an individual.
    selection_type: Literal[",", "+"]
        The type of selection to use for selecting new parents.
    """
    binary_population, population_f = initialize_population(problem=problem, initial_population_size=mu_)
    min_idx = np.argmin(population_f)
    f_opt, x_opt = population_f[min_idx], binary_population[min_idx]

    real_population = encode_to_real(binary_population=binary_population)
    n = real_population.shape[1]
    population_sigma = generate_initial_population_sigma(population=real_population, mutation_type=mutation_type,
                                                         initial_step_size=initial_step_size)
    population_rotation_angles = generate_rotation_angles(mutation_type=mutation_type,
                                                          defined_population_size=real_population.shape[0],
                                                          problem_dimension=n)
    while problem.state.evaluations < budget:
        binary_offspring, offspring_sigma, offspring_rotation_angles = recombination(
            lambda_=lambda_,
            binary_population=binary_population,
            population_sigma=population_sigma,
            population_rotation_angles=population_rotation_angles
        )
        real_offspring = encode_to_real(binary_population=binary_offspring)

        real_offspring, offspring_sigma, offspring_rotation_angles = mutation(
            mutation_type=mutation_type,
            population=real_offspring,
            population_sigma=offspring_sigma,
            population_rotation_angles=offspring_rotation_angles,
            problem_dimension=n
        )
        binary_offspring = decode_to_binary(real_population=real_offspring)
        offspring_f = []
        for i in range(lambda_):
            o = binary_offspring[i]
            f = problem(o)
            offspring_f.append(f)
            if f > f_opt:
                f_opt = f
                x_opt = o
            if problem.state.evaluations >= budget:
                break
        offspring_f = np.array(offspring_f)

        real_population, binary_population, population_sigma, population_rotation_angles, population_f = selection(
            selection_type=selection_type, real_population=real_population, binary_population=binary_population,
            real_offspring=real_offspring, binary_offspring=binary_offspring, population_sigma=population_sigma,
            offspring_sigma=offspring_sigma, population_f=population_f, offspring_f=offspring_f,
            population_rotation_angles=population_rotation_angles, offspring_rotation_angles=offspring_rotation_angles,
            mu_=mu_
        )


def create_problem(fid: int, root: str = "data", folder_name: str = "run_", algorithm_name: str = "ES",
                   algorithm_info: str = "Practical assignment of EA", timestamp: bool = True) -> Tuple[
    Union[ioh.iohcpp.problem.IsingRing, ioh.iohcpp.problem.LABS], ioh.iohcpp.logger.Analyzer]:
    """
    Create an ioh.iohcpp.problem and add a logger to it.

    Parameters
    ----------
    fid: int
        ID of the benchmark function to use.
    root: str, optional
        Folder in which to store the data. The default is "data".
    folder_name: str, optional
        Folder in which to store the data for this particular run. The
        default is "run_". A timestamp is appended to this name if timestamp
        is set to True.
    algorithm_name: str, optional
        Algorithm name to attach to the logger. The default is "GA".
    algorithm_info: str, optional
        Algorithm info to include with the logger. The default is "Practical
        assignment of EA".
    timestamp: bool, optional
        Whether to add a timestamp to the folder_name.

    Returns
    -------
    problem: ioh.iohcpp.problem
        Implementation-ready problem with attached logger.
    log: ioh.iohcpp.logger.Analyzer
        Logger attached to the problem.
    """
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output.
    # You should compress the folder 'run' and upload it to IOHanalyzer.
    if timestamp:
        time_str = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        folder_name = folder_name + "-" + time_str
    log = logger.Analyzer(
        # the working directory in which a folder named `folder_name`
        # (the next argument) will be created to store data
        root=root,
        folder_name=folder_name,  # the folder name to which the raw
        # performance
        # data will be stored
        algorithm_name=algorithm_name + "-" + time_str,  # name of your algorithm
        algorithm_info=algorithm_info,
    )
    # attach the logger to the problem
    problem.attach_logger(log)
    return problem, log


if __name__ == "__main__":
    # Run the algorithm with various settings
    mu_values = np.array([1, 2, 5, 10, 12, 15])
    lamdba_values = np.array([1, 2, 5, 10, 20, 50, 100])
    mutation_types = ["single", "individual", "correlated"]
    selection_types = [",", "+"]
    initial_step_sizes = [0.5, 1, 1.5, 2]
    hyperparameters = [
        {
            "mu": 12,
            "lambda": 50,
            "mutation_type": "correlated",
            "initial_step_size": 1,
            "selection_type": "+"
        },
        {
            "mu": 15,
            "lambda": 100,
            "mutation_type": "correlated",
            "initial_step_size": 0.5,
            "selection_type": ","
        },
        {
            "mu": 15,
            "lambda": 100,
            "mutation_type": "correlated",
            "initial_step_size": 1,
            "selection_type": ","
        },
        {
            "mu": 2,
            "lambda": 50,
            "mutation_type": "correlated",
            "initial_step_size": 1,
            "selection_type": "+"
        },
        {
            "mu": 12,
            "lambda": 100,
            "mutation_type": "correlated",
            "initial_step_size": 2,
            "selection_type": ","
        },
        {
            "mu": 15,
            "lambda": 100,
            "mutation_type": "correlated",
            "initial_step_size": 0.5,
            "selection_type": "+"
        },
        {
            "mu": 15,
            "lambda": 50,
            "mutation_type": "correlated",
            "initial_step_size": 0.5,
            "selection_type": "+"
        },
        {
            "mu": 10,
            "lambda": 20,
            "mutation_type": "correlated",
            "initial_step_size": 1,
            "selection_type": "+"
        },
        {
            "mu": 12,
            "lambda": 100,
            "mutation_type": "correlated",
            "initial_step_size": 0.5,
            "selection_type": ","
        },
    ]
    print(len(hyperparameters))
    n_runs = 20  # Nr of runs
    fid_values = [18, 19]  # Function IDs to run
    for fid in fid_values:
        for hyperparameter in hyperparameters:
            mu_ = hyperparameter["mu"]
            lambda_ = hyperparameter["lambda"]
            mutation_type = hyperparameter["mutation_type"]
            initial_step_size = hyperparameter["initial_step_size"]
            selection_type = hyperparameter["selection_type"]

            path = f"F{fid}-ES/({mu_}{selection_type}{lambda_})-ES-{mutation_type}-{initial_step_size}"
            print(path)
            Fid, _logger = create_problem(
                fid,
                folder_name=path,
                algorithm_name=f"({mu_}{selection_type}{lambda_})-ES-{mutation_type}-{initial_step_size}",
                algorithm_info=f"({mu_}{selection_type}{lambda_})-ES-{mutation_type}-{initial_step_size}")
            for run in range(n_runs):
                s4034538_s3366766_ES(Fid, lambda_=lambda_, mu_=mu_, mutation_type=mutation_type,
                                     selection_type=selection_type, initial_step_size=initial_step_size)
                Fid.reset()
            _logger.close()
