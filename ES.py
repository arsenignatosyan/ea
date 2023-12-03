from typing import Tuple, Union, List, Literal

import ioh
import numpy as np
from ioh import get_problem, logger, ProblemClass

np.random.seed(42)
budget = 5000
dimension = 50


def initialize_population(problem: Union[ioh.iohcpp.problem.IsingRing, ioh.iohcpp.problem.LABS],
                          initial_population_size: int) -> Tuple[np.ndarray, List[float]]:
    population = np.random.randint(2, size=(initial_population_size, problem.meta_data.n_variables))
    population_f = [problem(population_item) for population_item in population]

    return population, population_f


def calculate_initial_population_std(population: np.ndarray,
                                     mutation_type: Literal["single", "individual", "correlated"],
                                     problem_dimension: int = None) -> np.ndarray:
    population_sigma = np.std(population, axis=1)
    if mutation_type in ["individual", "correlated"]:
        sigma_rep = np.repeat(population_sigma.reshape(1, -1), problem_dimension, axis=0)
        population_sigma = sigma_rep.T

    return population_sigma


def generate_rotation_angles(mutation_type: Literal["single", "individual", "correlated"], defined_population_size: int,
                             problem_dimension: int) -> Union[np.ndarray, None]:
    rotation_angles = None
    if mutation_type == "correlated":
        k = int(problem_dimension * (problem_dimension - 1) / 2)
        rotation_angles = np.random.uniform(low=-np.pi, high=np.pi, size=(defined_population_size, k))

    return rotation_angles


def one_sigma_mutation(population: np.ndarray, population_sigma: np.ndarray, problem_dimension: int) -> Tuple[np.ndarray, np.ndarray]:
    tau = 1 / np.sqrt(problem_dimension)
    mut_sigma = [s * tau * np.exp(np.random.normal()) for s in population_sigma]
    mut_population = [p + s * np.random.normal() for p, s in zip(population, mut_sigma)]
    return np.array(mut_population), np.array(mut_sigma)


def individual_sigma_mutation(population: np.ndarray, population_sigma: np.ndarray, problem_dimension: int) -> Tuple[np.ndarray, np.ndarray]:
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
    count = 0
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            alpha_ij = alpha[count]
            mult_matrix = np.matmul(mult_matrix, rotation_matrix(alpha_ij=alpha_ij, i=i, j=j, n=n))
            count += 1

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


def mutation(mutation_type: Literal["single", "individual", "correlated"], population: np.ndarray,
             population_sigma: np.ndarray, population_rotation_angles: np.ndarray, problem_dimension: int) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
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


def discrete_recombination(parent: np.ndarray, parent_sigma: np.ndarray,
                           parent_rotation_angles: Union[np.ndarray, None]) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
    offspring, offspring_sigma = [], []
    offspring_rotation_angles = None if parent_rotation_angles is None else []
    sample_idxs = np.random.choice(len(parent), 2)
    sample_parent, sample_parent_sigma = parent[sample_idxs], parent_sigma[sample_idxs]
    sample_parent_rotation_angles = None if parent_rotation_angles is None else parent_rotation_angles[sample_idxs]
    for idx in range(parent.shape[1]):
        parent_idx = np.random.randint(len(sample_parent))
        offspring.append(parent[parent_idx][idx])
        offspring_sigma.append(parent_sigma[parent_idx][idx])

    if offspring_rotation_angles is not None:
        for angl_idx in range(sample_parent_rotation_angles.shape[1]):
            idx = np.random.randint(len(sample_parent))
            offspring_rotation_angles.append(sample_parent_rotation_angles[idx][angl_idx])

    return np.array(offspring), np.array(offspring_sigma), np.array(offspring_rotation_angles)


def intermediate_recombination(parent: np.ndarray, parent_sigma: np.ndarray,
                               parent_rotation_angles: Union[np.ndarray, None]) -> Tuple[
    np.ndarray, np.ndarray, Union[np.ndarray, None]]:
    sample_idxs = np.random.choice(len(parent), 2)
    sample_parent, sample_parent_sigma = parent[sample_idxs], parent_sigma[sample_idxs]
    sample_parent_rotation_angles = None if parent_rotation_angles is None else parent_rotation_angles[sample_idxs]
    offspring = np.mean(sample_parent, axis=0)
    offspring_sigma = np.mean(sample_parent_sigma, axis=0)
    offspring_rotation_angles = None if sample_parent_rotation_angles is None else np.mean(
        sample_parent_rotation_angles, axis=0)

    return offspring, offspring_sigma, offspring_rotation_angles


def global_discrete_recombination(parent: np.ndarray, parent_sigma: np.ndarray,
                                  parent_rotation_angles: Union[np.ndarray, None]) -> Tuple[
    np.ndarray, np.ndarray, Union[np.ndarray, None]]:
    offspring, offspring_sigma = [], []
    offspring_rotation_angles = None if parent_rotation_angles is None else []
    parent_idxs = []
    for idx in range(parent.shape[1]):
        parent_idx = np.random.randint(len(parent))
        parent_idxs.append(parent_idx)
        offspring.append(parent[parent_idx][idx])
        offspring_sigma.append(parent_sigma[parent_idx][idx])

    if offspring_rotation_angles is not None:
        for angl_idx in range(parent_rotation_angles.shape[1]):
            parent_idx = np.random.randint(len(parent_idxs))
            idx = parent_idxs[parent_idx]
            offspring_rotation_angles.append(parent_rotation_angles[idx][angl_idx])

    return np.array(offspring), np.array(offspring_sigma), np.array(offspring_rotation_angles)


def global_intermediate_recombination(parent: np.ndarray, parent_sigma: np.ndarray,
                                      parent_rotation_angles: Union[np.ndarray, None]) -> Tuple[
    np.ndarray, np.ndarray, Union[np.ndarray, None]]:
    offspring = np.mean(parent, axis=0)
    offspring_sigma = np.mean(parent_sigma, axis=0)
    offspring_rotation_angles = None if parent_rotation_angles is None else np.mean(parent_rotation_angles, axis=0)

    return offspring, offspring_sigma, offspring_rotation_angles


def recombination(population: np.ndarray, population_sigma: np.ndarray,
                  population_rotation_angles: Union[np.ndarray, None],
                  recombination_type: Literal["discrete", "intermediate", "global_discrete", "global_intermediate"]) -> \
        Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
    if recombination_type == "discrete":
        offspring, offspring_sigma, offspring_rotation_angles = discrete_recombination(
            parent=population,
            parent_sigma=population_sigma,
            parent_rotation_angles=population_rotation_angles
        )
    elif recombination_type == "intermediate":
        offspring, offspring_sigma, offspring_rotation_angles = intermediate_recombination(
            parent=population,
            parent_sigma=population_sigma,
            parent_rotation_angles=population_rotation_angles
        )
    elif recombination_type == "global_discrete":
        offspring, offspring_sigma, offspring_rotation_angles = global_discrete_recombination(
            parent=population,
            parent_sigma=population_sigma,
            parent_rotation_angles=population_rotation_angles
        )
    else:
        offspring, offspring_sigma, offspring_rotation_angles = global_intermediate_recombination(
            parent=population,
            parent_sigma=population_sigma,
            parent_rotation_angles=population_rotation_angles
        )

    return offspring, offspring_sigma, offspring_rotation_angles


def studentnumber1_studentnumber2_ES(problem: Union[ioh.iohcpp.problem.IsingRing, ioh.iohcpp.problem.LABS], mu_: int,
                                     lambda_: int, mutation_type: Literal["single", "individual", "correlated"],
                                     recombination_type: Literal[
                                         "discrete", "intermediate", "global_discrete", "global_intermediate"]):
    # hint: F18 and F19 are Boolean problems. Consider how to present bitstrings as real-valued vectors in ES
    # initial_pop = ... make sure you randomly create the first population
    population, population_f = initialize_population(problem=problem, initial_population_size=mu_)
    population_sigma = calculate_initial_population_std(population=initial_pop, mutation_type=mutation_type,
                                                        problem_dimension=problem_dimension)

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
        algorithm_name="ES",  # name of your algorithm
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
    pop, _ = initialize_population(problem=F18, initial_population_size=10)
    # print(pop)
    pop_std = calculate_initial_population_std(population=pop, mutation_type="correlated",
                                               problem_dimension=F18.meta_data.n_variables)
    rot_ang = generate_rotation_angles(defined_population_size=len(pop), problem_dimension=F18.meta_data.n_variables,
                                       mutation_type="correlated")
    # mut_pop, _, _ = correlated_mutation(population=pop, population_sigma=pop_std, population_rotation_angles=rot_ang,
    #                                     problem_dimension=F18.meta_data.n_variables)
    # mut_pop, _ = individual_sigma_mutation(population=pop, population_sigma=pop_std,
    #                                        problem_dimension=F18.meta_data.n_variables)
    mut_population_, mut_sigma_, mut_rotation_angles_ = mutation(mutation_type="correlated", population=pop,
                                                              population_sigma=pop_std,
                                                              population_rotation_angles=rot_ang,
                                                              problem_dimension=F18.meta_data.n_variables)
    print(pop.shape, pop_std.shape, rot_ang.shape)
    print(mut_population_.shape, mut_sigma_.shape, mut_rotation_angles_.shape)

    o, o_s, o_a = recombination(population=pop, population_sigma=pop_std,
                                population_rotation_angles=rot_ang, recombination_type="global_intermediate")
    print(o.shape, o_s.shape, o_a.shape)
    print("========================")
