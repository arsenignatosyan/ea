from typing import Tuple, Union, List, Literal

import ioh
import numpy as np
from ioh import get_problem, logger, ProblemClass

np.random.seed(42)
budget = 5000
dimension = 50
bit_chunk = 5


def initialize_population(problem: Union[ioh.iohcpp.problem.IsingRing, ioh.iohcpp.problem.LABS],
                          initial_population_size: int) -> Tuple[np.ndarray[bool], List[float]]:
    population = np.random.randint(2, size=(initial_population_size, problem.meta_data.n_variables))
    population_f = [problem(population_item) for population_item in population]

    return population, population_f


def calculate_initial_population_std(population: np.ndarray[float],
                                     mutation_type: Literal["single", "individual", "correlated"],
                                     problem_dimension: int = None) -> np.ndarray[float]:
    population_sigma = np.std(population, axis=1)
    if mutation_type in ["individual", "correlated"]:
        sigma_rep = np.repeat(population_sigma.reshape(1, -1), problem_dimension, axis=0)
        population_sigma = sigma_rep.T

    return population_sigma


def generate_rotation_angles(mutation_type: Literal["single", "individual", "correlated"], defined_population_size: int,
                             problem_dimension: int) -> Union[np.ndarray[float], None]:
    rotation_angles = None
    if mutation_type == "correlated":
        k = int(problem_dimension * (problem_dimension - 1) / 2)
        rotation_angles = np.random.uniform(low=-np.pi, high=np.pi, size=(defined_population_size, k))

    return rotation_angles


def one_sigma_mutation(population: np.ndarray[float], population_sigma: np.ndarray[float], problem_dimension: int) -> \
        Tuple[
            np.ndarray[float], np.ndarray[float]]:
    tau = 1 / np.sqrt(problem_dimension)
    mut_sigma = [s * tau * np.exp(np.random.normal()) for s in population_sigma]
    mut_population = [p + s * np.random.normal() for p, s in zip(population, mut_sigma)]
    return np.array(mut_population), np.array(mut_sigma)


def individual_sigma_mutation(population: np.ndarray[float], population_sigma: np.ndarray[float],
                              problem_dimension: int) -> Tuple[
    np.ndarray[float], np.ndarray[float]]:
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
    R = np.eye(n)
    cos_alpha, sin_alpha = np.cos(alpha_ij), np.sin(alpha_ij)

    R[i, i] = cos_alpha
    R[j, j] = cos_alpha
    R[i, j] = -sin_alpha
    R[j, i] = sin_alpha

    return R


def multiply_rotation_matrix(alpha: np.ndarray[float], n: int) -> np.ndarray[float]:
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


def mutation(mutation_type: Literal["single", "individual", "correlated"], population: np.ndarray[float],
             population_sigma: np.ndarray[float], population_rotation_angles: np.ndarray[float],
             problem_dimension: int) -> Tuple[
    np.ndarray[float], np.ndarray[float], Union[np.ndarray[float], None]]:
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


def discrete_recombination(parent: np.ndarray[bool], parent_sigma: np.ndarray[float],
                           parent_rotation_angles: Union[np.ndarray[float], None]) -> Tuple[
    np.ndarray[bool], np.ndarray[float], Union[np.ndarray[float], None]]:
    offspring, offspring_sigma = [], []
    offspring_rotation_angles = None if parent_rotation_angles is None else []
    sample_idxs = np.random.choice(len(parent), 2)
    sample_parent, sample_parent_sigma = parent[sample_idxs], parent_sigma[sample_idxs]
    sample_parent_rotation_angles = None if parent_rotation_angles is None else parent_rotation_angles[sample_idxs]
    for idx in range(parent.shape[1]):
        parent_idx = np.random.randint(len(sample_parent))
        offspring.append(sample_parent[parent_idx][idx])

    for idx in range(parent_sigma.shape[1]):
        parent_idx = np.random.randint(len(sample_parent))
        offspring_sigma.append(sample_parent_sigma[parent_idx][idx])

    if offspring_rotation_angles is not None:
        for angl_idx in range(sample_parent_rotation_angles.shape[1]):
            parent_idx = np.random.randint(len(sample_parent))
            offspring_rotation_angles.append(sample_parent_rotation_angles[parent_idx][angl_idx])

    return np.array(offspring), np.array(offspring_sigma), np.array(offspring_rotation_angles)


def global_discrete_recombination(parent: np.ndarray[bool], parent_sigma: np.ndarray[float],
                                  parent_rotation_angles: Union[np.ndarray[float], None]) -> Tuple[
    np.ndarray[bool], np.ndarray[float], Union[np.ndarray[float], None]]:
    offspring, offspring_sigma = [], []
    offspring_rotation_angles = None if parent_rotation_angles is None else []
    parent_idxs = []
    for idx in range(parent.shape[1]):
        parent_idx = np.random.randint(len(parent))
        parent_idxs.append(parent_idx)
        offspring.append(parent[parent_idx][idx])

    for idx in range(parent_sigma.shape[1]):
        parent_idx = parent_idxs[np.random.randint(len(parent_idxs))]
        offspring_sigma.append(parent_sigma[parent_idx][idx])

    if offspring_rotation_angles is not None:
        for angl_idx in range(parent_rotation_angles.shape[1]):
            parent_idx = parent_idxs[np.random.randint(len(parent_idxs))]
            offspring_rotation_angles.append(parent_rotation_angles[parent_idx][angl_idx])

    return np.array(offspring), np.array(offspring_sigma), np.array(offspring_rotation_angles)


def intermediate_recombination(parent: np.ndarray[float], parent_sigma: np.ndarray[float],
                               parent_rotation_angles: Union[np.ndarray[float], None]) -> Tuple[
    np.ndarray[bool], np.ndarray[float], Union[np.ndarray[float], None]]:
    sample_idxs = np.random.choice(len(parent), 2)
    sample_parent, sample_parent_sigma = parent[sample_idxs], parent_sigma[sample_idxs]
    sample_parent_rotation_angles = None if parent_rotation_angles is None else parent_rotation_angles[sample_idxs]
    offspring = np.mean(sample_parent, axis=0)
    offspring_sigma = np.mean(sample_parent_sigma, axis=0)
    offspring_rotation_angles = None if sample_parent_rotation_angles is None else np.mean(
        sample_parent_rotation_angles, axis=0)

    return offspring, offspring_sigma, offspring_rotation_angles


def global_intermediate_recombination(parent: np.ndarray[float], parent_sigma: np.ndarray[float],
                                      parent_rotation_angles: Union[np.ndarray[float], None]) -> Tuple[
    np.ndarray[float], np.ndarray[float], Union[np.ndarray[float], None]]:
    offspring = np.mean(parent, axis=0)
    offspring_sigma = np.mean(parent_sigma, axis=0)
    offspring_rotation_angles = None if parent_rotation_angles is None else np.mean(parent_rotation_angles, axis=0)

    return offspring, offspring_sigma, offspring_rotation_angles


def recombination(lambda_: int, binary_population: np.ndarray[bool], real_population: np.ndarray[float],
                  population_sigma: np.ndarray[float], population_rotation_angles: Union[np.ndarray[float], None],
                  recombination_type: Literal["discrete", "intermediate", "global_discrete", "global_intermediate"]) -> \
        Tuple[np.ndarray[float], np.ndarray[float], Union[np.ndarray[float], None]]:
    offspring, offspring_sigma, offspring_rotation_angles = [], [], []
    for _ in lambda_:
        if recombination_type == "discrete":
            o, o_sigma, o_rotation_angles = discrete_recombination(
                parent=binary_population,
                parent_sigma=population_sigma,
                parent_rotation_angles=population_rotation_angles
            )
            o = encode_binary(binary_population=o)
        elif recombination_type == "intermediate":
            o, o_sigma, o_rotation_angles = intermediate_recombination(
                parent=real_population,
                parent_sigma=population_sigma,
                parent_rotation_angles=population_rotation_angles
            )
        elif recombination_type == "global_discrete":
            o, o_sigma, o_rotation_angles = global_discrete_recombination(
                parent=binary_population,
                parent_sigma=population_sigma,
                parent_rotation_angles=population_rotation_angles
            )
            offspring = encode_binary(binary_population=offspring)
        else:
            o, o_sigma, o_rotation_angles = global_intermediate_recombination(
                parent=real_population,
                parent_sigma=population_sigma,
                parent_rotation_angles=population_rotation_angles
            )
        offspring.append(o)
        offspring_sigma.append(o_sigma)
        offspring_rotation_angles.append(o_rotation_angles)

    return np.array(offspring), np.array(offspring_sigma), np.array(offspring_rotation_angles)


def encode_binary(binary_population: np.ndarray[bool]) -> np.ndarray[int]:
    real_population = []
    for bit_vector in binary_population:
        bit_vector = [str(b) for b in bit_vector]
        bit_vector_str = ''.join(bit_vector)
        real_vector = [int(bit_vector_str[i:i + bit_chunk], 2) for i in range(0, len(bit_vector), bit_chunk)]
        real_population.append(real_vector)

    return np.array(real_population)


def decode_binary(real_population: np.ndarray[float]) -> np.ndarray[bool]:
    real_population = real_population.round().astype(int)

    binary_population = []
    for i, real_vector in enumerate(real_population):
        for j in range(len(real_vector)):
            value_j = real_vector[j]
            real_vector[j] = 31 if value_j >= pow(2, bit_chunk) else value_j
            real_vector[j] = 0 if value_j < 0 else value_j
        bit_vector = ['{0:05b}'.format(r) for r in real_vector]
        bit_vector = [[*b] for b in bit_vector]
        bit_vector_flat = [int(item) for sublist in bit_vector for item in sublist]
        binary_population.append(bit_vector_flat)

    return np.array(binary_population)


def selection(selection_type: Literal[",", "+"], real_population: np.ndarray[float],
              binary_population: np.ndarray[bool],
              real_offspring: np.ndarray[float], binary_offspring: np.ndarray[bool],
              population_sigma: np.ndarray[float],
              offspring_sigma: np.ndarray[float], population_rotation_angles: np.ndarray[float],
              offspring_rotation_angles: np.ndarray[float],
              population_f: np.ndarray[float], offspring_f: np.ndarray[float],
              mu_: int) -> Tuple[
    np.ndarray[float], np.ndarray[bool], np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
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
        considered_rotation_angles = np.append(arr=population_rotation_angles, values=offspring_rotation_angles, axis=0)
        considered_f = np.append(arr=population_f, values=offspring_f, axis=0)

    min_idx = np.argsort(considered_f)[:mu_]
    new_real = considered_real[min_idx]
    new_binary = considered_binary[min_idx]
    new_sigma = considered_sigma[min_idx]
    new_rotation_angles = considered_rotation_angles[min_idx]
    new_f = considered_f[min_idx]

    return new_real, new_binary, new_sigma, new_rotation_angles, new_f


def studentnumber1_studentnumber2_ES(problem: Union[ioh.iohcpp.problem.IsingRing, ioh.iohcpp.problem.LABS], mu_: int,
                                     lambda_: int, mutation_type: Literal["single", "individual", "correlated"],
                                     recombination_type: Literal[
                                         "discrete", "intermediate", "global_discrete", "global_intermediate"],
                                     selection_type: Literal[",", "+"]):
    # hint: F18 and F19 are Boolean problems. Consider how to present bitstrings as real-valued vectors in ES
    binary_population, population_f = initialize_population(problem=problem, initial_population_size=mu_)
    min_idx = np.argmin(population_f)
    f_opt, x_opt = population_f[min_idx], binary_population[min_idx]

    real_population = encode_binary(binary_population=binary_population)
    n = real_population.shape[1]
    population_sigma = calculate_initial_population_std(population=real_population, mutation_type=mutation_type,
                                                        problem_dimension=n)
    population_rotation_angles = generate_rotation_angles(mutation_type=mutation_type,
                                                          defined_population_size=real_population.shape[0],
                                                          problem_dimension=n)

    while problem.state.evaluations < budget:
        real_offspring, offspring_sigma, offspring_rotation_angles = recombination(
            lambda_=lambda_,
            binary_population=binary_population,
            real_population=real_population,
            population_sigma=population_sigma,
            recombination_type=recombination_type,
            population_rotation_angles=population_rotation_angles
        )

        real_offspring, offspring_sigma, offspring_rotation_angles = mutation(
            mutation_type=mutation_type,
            population=real_offspring,
            population_sigma=offspring_sigma,
            population_rotation_angles=offspring_rotation_angles,
            problem_dimension=n
        )
        binary_offspring = decode_binary(real_population=real_offspring)
        offspring_f = []
        for i in range(lambda_):
            o = binary_offspring[i]
            f = problem(o)
            offspring_f.append(f)
            if f < f_opt:
                f_opt = f
                x_opt = o
            if problem.state.evaluations < budget:
                break

        real_population, binary_population, population_sigma, population_rotation_angles, population_f = selection(
            selection_type=selection_type, real_population=real_population,
            binary_population=binary_population, real_offspring=real_offspring, binary_offspring=binary_offspring,
            population_sigma=population_sigma,
            offspring_sigma=offspring_sigma, population_rotation_angles=population_rotation_angles,
            population_f=population_f, offspring_f=offspring_f,
            offspring_rotation_angles=offspring_rotation_angles, mu_=mu_
        )

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
    print(pop)
    int_ = encode_binary(pop)
    print(int_)
    bit_ = decode_binary(real_population=int_)
    print(bit_)
    # pop_std = calculate_initial_population_std(population=pop, mutation_type="correlated",
    #                                            problem_dimension=F18.meta_data.n_variables)
    # rot_ang = generate_rotation_angles(defined_population_size=len(pop), problem_dimension=F18.meta_data.n_variables,
    #                                    mutation_type="correlated")
    # # mut_pop, _, _ = correlated_mutation(population=pop, population_sigma=pop_std, population_rotation_angles=rot_ang,
    # #                                     problem_dimension=F18.meta_data.n_variables)
    # # mut_pop, _ = individual_sigma_mutation(population=pop, population_sigma=pop_std,
    # #                                        problem_dimension=F18.meta_data.n_variables)
    # mut_population_, mut_sigma_, mut_rotation_angles_ = mutation(mutation_type="correlated", population=pop,
    #                                                           population_sigma=pop_std,
    #                                                           population_rotation_angles=rot_ang,
    #                                                           problem_dimension=F18.meta_data.n_variables)
    # print(pop.shape, pop_std.shape, rot_ang.shape)
    # print(mut_population_.shape, mut_sigma_.shape, mut_rotation_angles_.shape)
    #
    # o, o_s, o_a = recombination(population=pop, population_sigma=pop_std,
    #                             population_rotation_angles=rot_ang, recombination_type="global_intermediate")
    # print(o.shape, o_s.shape, o_a.shape)
    # print("========================")
