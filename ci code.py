import random
import numpy as np

def objective_function(solution):
    # Define your objective function here
    # It should take a solution (resource assignment) as input and return a scalar value (e.g., total cost, fitness)
    # You can customize this function based on your specific problem

    # Example objective function: minimizing the total cost
    total_cost = np.sum(solution)
    return total_cost

def initialize_population(pop_size, num_schools):
    population = []
    for _ in range(pop_size):
        solution = [random.randint(0, num_schools-1) for _ in range(num_students)]
        population.append(solution)
    return population

def differential_evolution(objective_func, bounds, pop_size, max_generations, F=0.5, CR=0.9):
    num_students = len(bounds)
    num_schools = len(set(bounds))

    population = initialize_population(pop_size, num_schools)
    best_solution = None
    best_fitness = float('inf')

    for _ in range(max_generations):
        for i in range(pop_size):
            target = population[i]
            a, b, c = random.sample(population, 3)
            mutant = np.clip(a + F * (b - c), bounds[0], bounds[1])
            trial = np.where(np.random.rand(num_students) < CR, mutant, target)

            fitness = objective_func(trial)
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = trial

            population[i] = trial

    return best_solution
# Example usage
num_students = 100
bounds = (0, 9)  # Assign students to schools numbered from 0 to 9
pop_size = 50
max_generations = 100

best_assignment = differential_evolution(objective_function, bounds, pop_size, max_generations)
print("Best assignment:", best_assignment)
def objective_function(solution):
    # Define your objective function here
    # It should take a solution (resource assignment) as input and return a scalar value (e.g., total cost, fitness)
    # You can customize this function based on your specific problem

    # Example objective function: minimizing the total cost
    total_cost = np.sum(solution)
    return total_cost

def initialize_population(pop_size, num_schools):
    population = []
    for _ in range(pop_size):
        solution = [random.randint(0, num_schools-1) for _ in range(num_students)]
        population.append(solution)
    return population

def differential_evolution(objective_func, bounds, pop_size, max_generations, F=0.5, CR=0.9):
    num_students = len(bounds)
    num_schools = len(set(bounds))

    population = initialize_population(pop_size, num_schools)
    best_solution = None
    best_fitness = float('inf')

    for _ in range(max_generations):
        for i in range(pop_size):
            target = population[i]
            a, b, c = random.sample(population, 3)
            mutant = np.clip(a + F * (b - c), bounds[0], bounds[1])
            trial = np.where(np.random.rand(num_students) < CR, mutant, target)

            fitness = objective_func(trial)
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = trial

            population[i] = trial

    return best_solution
# Example usage
num_students = 100
bounds = (0, 9)  # Assign students to schools numbered from 0 to 9
pop_size = 50
max_generations = 100

best_assignment = differential_evolution(objective_function, bounds, pop_size, max_generations)
print("Best assignment:", best_assignment)
print("Best fitness:", objective_function(best_assignment))
