import random
from typing import Tuple, Union, List, Literal

# you need to install this package `ioh`. Please see documentations here:
# https://iohprofiler.github.io/IOHexp/ and
# https://pypi.org/project/ioh/
import ioh
import numpy as np
from ioh import get_problem, logger, ProblemClass

initial_budget = 5000
dimension = 50
population_size = 10


# To make your results reproducible (not required by the assignment), you could set the random seed by
# `np.random.seed(some integer, e.g., 42)`

def initialize_population(problem: Union[ioh.iohcpp.problem.IsingRing, ioh.iohcpp.problem.LABS], budget: int,
                          initial_population_size: int = population_size) -> Tuple[np.ndarray, List[float], int]:
    population = np.random.randint(2, size=(initial_population_size, problem.meta_data.n_variables))
    population_f = [problem(population_item) for population_item in population]
    budget -= initial_population_size

    return population, population_f, budget


def calculate_initial_population_std(population: np.ndarray,
                                     mutation_type: Literal["single", "individual", "correlated"],
                                     problem_dimension: int = None) -> np.ndarray:
    population_sigma = np.std(population, axis=1)
    if mutation_type in ["individual", "correlated"]:
        sigma_rep = np.repeat(population_sigma.reshape(1, -1), problem_dimension, axis=0)
        population_sigma = sigma_rep.T

    return population_sigma


def generate_rotation_angles(defined_population_size: int, problem_dimension: int) -> np.ndarray:
    k = int(problem_dimension * (problem_dimension - 1) / 2)
    rotation_angles = np.random.uniform(low=-np.pi, high=np.pi, size=(defined_population_size, k))
    return rotation_angles


def one_sigma_mutation(population: np.ndarray, population_sigma: np.ndarray, problem_dimension: int) -> Tuple[
    np.ndarray, np.ndarray]:
    tau = 1 / np.sqrt(problem_dimension)
    mut_sigma = [s * tau * np.exp(np.random.normal()) for s in population_sigma]
    mut_population = [p + s * np.random.normal() for p, s in zip(population, mut_sigma)]
    return np.array(mut_population), np.array(mut_sigma)


def individual_sigma_mutation(population: np.ndarray, population_sigma: np.ndarray, problem_dimension: int) -> Tuple[
    np.ndarray, np.ndarray]:
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


def rotation_matrix(alpha_ij: float, i: int, j: int, n: int) -> np.ndarray:
    R = np.eye(n)
    cos_alpha, sin_alpha = np.cos(alpha_ij), np.sin(alpha_ij)

    R[i, i] = cos_alpha
    R[j, j] = cos_alpha
    R[i, j] = -sin_alpha
    R[j, i] = sin_alpha

    return R


def multiply_rotation_matrix(alpha: np.ndarray, n: int) -> np.ndarray:
    mult_matrix = np.eye(n)
    for i in range(n - 1):
        for j in range(i, n):
            alpha_ij = alpha[i + j]
            mult_matrix = np.matmul(mult_matrix, rotation_matrix(alpha_ij=alpha_ij, i=i, j=j, n=n))

    return mult_matrix


def correlated_mutation(population: np.ndarray, population_sigma: np.ndarray, population_rotation_angles: np.ndarray,
                        problem_dimension: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tau = 1 / np.sqrt(2 * problem_dimension)
    tau_prime = 1 / np.sqrt(2 * np.sqrt(problem_dimension))
    g = np.random.normal()
    beta = np.pi / 36
    mut_population = np.empty(population.shape)
    mut_sigma = np.empty(population_sigma.shape)
    mut_rotation_angles = np.empty(population_rotation_angles.shape)
    for i in range(population.shape[0]):
        for j in range(population.shape[1]):
            mut_sigma[i][j] = population_sigma[i][j] * np.exp(tau_prime * g + tau * np.random.normal())

        for k in range(population_rotation_angles.shape[1]):
            mut_rotation_angles[i][k] = population_rotation_angles[i][k] + np.random.normal(0, beta)
            if abs(mut_rotation_angles[i][k]) > np.pi:
                mut_rotation_angles[i][k] = mut_rotation_angles[i][k] - 2 * np.pi * np.sign(mut_rotation_angles[i][k])

        sigma_mat = np.eye(problem_dimension) * mut_sigma[i]
        mult_rotation_matrix = multiply_rotation_matrix(alpha=mut_rotation_angles[i], n=problem_dimension)
        C_sqrt = np.matmul(mult_rotation_matrix, sigma_mat)
        C_prime = np.matmul(C_sqrt, C_sqrt.T)
        mut_population[i] = population[i] + np.random.multivariate_normal(mean=np.zeros(problem_dimension), cov=C_prime)

    return mut_population, mut_sigma, mut_rotation_angles


# Recombination
def discrete_recombination(parent: np.ndarray) -> np.ndarray:
    sample_parents = parent[np.random.choice(len(parent), 2)]
    parent1, parent2 = sample_parents[0], sample_parents[1]
    offspring = []
    for x1, x2 in zip(parent1, parent2):
        offspring.append(random.choice([x1, x2]))

    return np.array(offspring)


def intermediate_recombination(parent: np.ndarray) -> np.ndarray:
    sample_parents = parent[np.random.choice(len(parent), 2)]
    parent1, parent2 = sample_parents[0], sample_parents[1]
    offspring = []
    for x1, x2 in zip(parent1, parent2):
        offspring.append([x1, x2])

    return np.array(offspring)


def global_discrete_recombination(parent: np.ndarray) -> np.ndarray:
    offspring = []
    for idx in range(parent.shape[1]):
        parent_idx = np.random.randint(len(parent))
        offspring.append(parent[parent_idx][idx])

    return np.array(offspring)


def global_intermediate_recombination(parent: np.ndarray) -> np.ndarray:
    offspring = np.mean(parent, axis=0)

    return offspring


def studentnumber1_studentnumber2_ES(problem: Union[ioh.iohcpp.problem.IsingRing, ioh.iohcpp.problem.LABS]):
    # hint: F18 and F19 are Boolean problems. Consider how to present bitstrings as real-valued vectors in ES
    # initial_pop = ... make sure you randomly create the first population
    initial_pop, initial_pop_f, budget = initialize_population(problem=problem, budget=initial_budget)

    # `problem.state.evaluations` counts the number of function evaluation automatically,
    # which is incremented by 1 whenever you call `problem(x)`.
    # You could also maintain a counter of function evaluations if you prefer.
    while problem.state.evaluations < budget:
        # please implement the mutation, crossover, selection here
        # .....
        # this is how you evaluate one solution `x`
        # f = problem(x)
        pass
    # no return value needed


def create_problem(fid: int) -> Tuple[Union[ioh.iohcpp.problem.IsingRing, ioh.iohcpp.problem.LABS],
                                      ioh.iohcpp.logger.Analyzer]:
    # Declaration of problems to be tested.
    problem = get_problem(fid, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)

    # Create default logger compatible with IOHanalyzer
    # `root` indicates where the output files are stored.
    # `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
    l = logger.Analyzer(
        root="data",
        # the working directory in which a folder named `folder_name` (the next argument) will be created to store data
        folder_name="run",  # the folder name to which the raw performance data will be stored
        algorithm_name="evolution_strategy",  # name of your algorithm
        algorithm_info="Practical assignment of the EA course",
    )
    # attach the logger to the problem
    problem.attach_logger(l)
    return problem, l


# if __name__ == "__main__":
#     # this how you run your algorithm with 20 repetitions/independent run
#     F18, _logger = create_problem(18)
#     for run in range(20):
#         studentnumber1_studentnumber2_ES(F18)
#         F18.reset()  # it is necessary to reset the problem after each independent run
#     _logger.close()  # after all runs, it is necessary to close the logger to make sure all data are written to the folder
#
#     F19, _logger = create_problem(19)
#     for run in range(20):
#         studentnumber1_studentnumber2_ES(F19)
#         F19.reset()
#     _logger.close()

if __name__ == "__main__":
    F18, _logger = create_problem(18)
    pop, _, _ = initialize_population(problem=F18, budget=300)
    # print(pop)
    pop_std = calculate_initial_population_std(population=pop, mutation_type="correlated",
                                               problem_dimension=F18.meta_data.n_variables)
    rot_ang = generate_rotation_angles(defined_population_size=len(pop), problem_dimension=F18.meta_data.n_variables)
    mut_pop, _, _ = correlated_mutation(population=pop, population_sigma=pop_std, population_rotation_angles=rot_ang,
                                        problem_dimension=F18.meta_data.n_variables)
    # mut_pop, _ = individual_sigma_mutation(population=pop, population_sigma=pop_std,
    #                                        problem_dimension=F18.meta_data.n_variables)
    print(mut_pop)
    print("========================")
