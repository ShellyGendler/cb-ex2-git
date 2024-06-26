import random
import numpy as np
import matplotlib.pyplot as plt

def read_prefs(file_path):
    # read all lines
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Calculate the total number of lines and midpoint
    total_lines = len(lines)
    midpoint = total_lines // 2
    
    # Initialize preferences
    men_prefs = []
    women_prefs = []
    
    # Split lines into men's and women's preferences
    for i in range(midpoint):
        prefs = list(map(int, lines[i].strip().split()))
        men_prefs.append(prefs)
    
    for i in range(midpoint, total_lines):
        prefs = list(map(int, lines[i].strip().split()))
        women_prefs.append(prefs)
    
    return np.array(men_prefs), np.array(women_prefs), total_lines

# Fitness function
def fitness(solution, men_prefs, women_prefs, total_lines):
    blocking_pairs = 0
    n = len(solution)
    # Iterate over each man in the solution
    for man in range(n):
        woman = solution[man]
        # Check for blocking pairs with every other man
        for other_man in range(n):
            if other_man != man:
                other_woman = solution[other_man]
                # Check if there is a blocking pair
                if (np.where(men_prefs[man] == woman+1)[0][0] > np.where(men_prefs[man] == other_woman+1)[0][0] and
                    np.where(women_prefs[other_woman] == man+1)[0][0] > np.where(women_prefs[other_woman] == other_man+1)[0][0]):
                    blocking_pairs += 1
     # number of "happy" couples               
    return total_lines // 2 - blocking_pairs

# Order Crossover
def order_crossover(parent1, parent2):
    size = len(parent1)
   
    # Choose two random crossover points
    start, end = sorted(random.sample(range(size), 2))
   
    # Create placeholders for the children
    child1 = [-1] * size
    child2 = [-1] * size
   
    # Copy the segment from parent1 to child1 and from parent2 to child2
    child1[start:end] = parent1[start:end]
    child2[start:end] = parent2[start:end]
   
    # Fill the remaining positions
    def fill_remaining_positions(child, parent):
        pos = end
        for i in range(size):
            if parent[(end + i) % size] not in child:
                child[pos % size] = parent[(end + i) % size]
                pos += 1

    fill_remaining_positions(child1, parent2)
    fill_remaining_positions(child2, parent1)
   
    return child1, child2

# Mutation: swap elements
def mutation(solution, mutation_rate=0.01):
    # Perform mutation with a given probability
    if random.random() < mutation_rate:
        # Select 3 positions to swap
        a, b, c = random.sample(range(len(solution)), 3)
        # Swap the selected positions
        solution[a], solution[b], solution[c] = solution[c], solution[a], solution[b]
    return solution

def selection(population, fitnesses, crowding_distances, k=3):
    # Randomly select k individuals and return the best one considering both fitness and crowding distance
    selected = random.choices(list(range(len(population))), k=k)
    selected_fitnesses = [fitnesses[i] for i in selected]
    best_index = selected[np.argmin(selected_fitnesses)]
   
    # In case of ties in fitness, use crowding distance to decide
    best_fitness = fitnesses[best_index]
    candidates = [i for i in selected if fitnesses[i] == best_fitness]
   
    if len(candidates) > 1:
        best_index = max(candidates, key=lambda i: crowding_distances[i])
   
    return population[best_index]

# Preserve diversity: crowding distance calculation
def calculate_crowding_distance(fitnesses):
    crowding_distances = [0] * len(fitnesses)
    sorted_indices = np.argsort(fitnesses)
   
    # Set the boundary points' crowding distance to a large number
    crowding_distances[sorted_indices[0]] = float('inf')
    crowding_distances[sorted_indices[-1]] = float('inf')
   
    # Calculate crowding distance for each individual
    for i in range(1, len(fitnesses) - 1):
        crowding_distances[sorted_indices[i]] += (fitnesses[sorted_indices[i+1]] - fitnesses[sorted_indices[i-1]])
   
    return crowding_distances

# Genetic Algorithm
def genetic_algorithm(men_prefs, women_prefs, total_lines, pop_size=30, max_generations=600, mutation_rate=0.01):
    # Initialize the population with random solutions
    population = [random.sample(range(total_lines // 2), total_lines // 2) for _ in range(pop_size)]
    best_solution = None
    best_fitness = float('inf')
    
    # To store fitness values over generations
    best_fitness_over_gens = []
    worst_fitness_over_gens = []
    average_fitness_over_gens = []
   
    # Iterate through generations
    for generation in range(max_generations):
        fitnesses = [fitness(ind, men_prefs, women_prefs, total_lines) for ind in population]
        new_population = []
        crowding_distance = calculate_crowding_distance(fitnesses)
        # Generate new population through crossover and mutation
        for _ in range(pop_size // 2):
            parent1 = selection(population, fitnesses, crowding_distance)
            parent2 = selection(population, fitnesses, crowding_distance)
            child1, child2 = order_crossover(parent1, parent2)
            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)
            new_population.extend([child1, child2])
       
        population = new_population
        current_best_fitness = max(fitnesses)
        current_worst_fitness = min(fitnesses)
        current_average_fitness = np.mean(fitnesses)

        current_best_solution = population[fitnesses.index(current_best_fitness)]
        print("current best solution\n")
        print(current_best_solution)
        
        # Record fitness values for the current generation
        best_fitness_over_gens.append(current_best_fitness)
        worst_fitness_over_gens.append(current_worst_fitness)
        average_fitness_over_gens.append(current_average_fitness)
       
        # Update the best solution found
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            print("best fitness\n")
            print(best_fitness)
            best_solution = current_best_solution
            print("new pop\n")
            print(best_solution)
       
        # Early stopping if a perfect solution is found
        if best_fitness == total_lines // 2:
            break
   
    return best_solution, best_fitness, best_fitness_over_gens, worst_fitness_over_gens, average_fitness_over_gens
# main
# Run the genetic algorithm
# Initialize preferences
file_path = r"C:\Users\Admin\Downloads\GA_input.txt"
men_prefs, women_prefs, total_lines = read_prefs(file_path)
print("Men's Preferences:")
print(men_prefs)
print("\nWomen's Preferences:")
print(women_prefs)

best_solution, best_fitness, best_fitness_over_gens, worst_fitness_over_gens, average_fitness_over_gens = genetic_algorithm(men_prefs, women_prefs, total_lines)
print(f"Best Solution: {best_solution}")
print(f"Best Fitness: {best_fitness}")

# Plotting the fitness values over generations
plt.figure(figsize=(12, 6))
plt.plot(best_fitness_over_gens, label='Best Fitness')
plt.plot(worst_fitness_over_gens, label='Worst Fitness')
plt.plot(average_fitness_over_gens, label='Average Fitness')
plt.xlabel('Generations')
plt.ylabel('Fitness Value')
plt.title('Fitness Value over Generations')
plt.legend()
plt.grid(True)
plt.show()