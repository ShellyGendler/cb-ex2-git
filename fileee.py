import random
import numpy as np

# Reading preferences from file
def read_prefs(file_path):
    men_prefs = []
    women_prefs = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
        # First 30 lines for men's preferences
        for i in range(30):
            prefs = list(map(int, lines[i].strip().split()))
            men_prefs.append(prefs)
        
        # Next 30 lines for women's preferences
        for i in range(30, 60):
            prefs = list(map(int, lines[i].strip().split()))
            women_prefs.append(prefs)
    
    return np.array(men_prefs), np.array(women_prefs)


# Fitness function
def fitness(solution):
    blocking_pairs = 0
    # Iterate over each man in the solution
    for i in range(len(solution)):
        man = i
        woman = solution[i]
        # Check for blocking pairs with every other man
        for j in range(len(solution)):
            if j != i:
                other_man = j
                other_woman = solution[j]
                # Check if there is a blocking pair
                if (np.where(men_prefs[man] == woman+1)[0][0] > np.where(men_prefs[man] == other_woman+1)[0][0] and 
                    np.where(women_prefs[other_woman] == man+1)[0][0] > np.where(women_prefs[other_woman] == other_man+1)[0][0]):
                    blocking_pairs += 1
    return blocking_pairs

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
def mutation(solution, mutation_rate=0.1):
    # Perform mutation with a given probability
    if random.random() < mutation_rate:
        # Select 7 positions to swap
        a, b, c, d, e, f, g = random.sample(range(len(solution)), 7)
        # Swap the selected positions
        solution[a], solution[b], solution[c], solution[d], solution[e], solution[f], solution[g] = (
            solution[g], solution[f], solution[e], solution[d], solution[c], solution[b], solution[a]
        )
    return solution

# Genetic Algorithm
def genetic_algorithm(pop_size=100, max_generations=100, mutation_rate=0.1):
    # Initialize the population with random solutions (random pairs)
    # 100 arrays of 30 people
    population = [random.sample(range(30), 30) for _ in range(pop_size)]
    best_solution = None
    best_fitness = float('inf')
    
    # Iterate through generations
    for generation in range(max_generations):
        # call fitness function
        # array of grades - num of blocking pairs, the array size is 100
        fitnesses = [fitness(ind) for ind in population]
        # the new population is made of the children only
        new_population = []
        
        # Generate new population through crossover and mutation
        for _ in range(pop_size // 2):
            # choose the best one to be a parent - once for men and for women
            parent1 = selection(population, fitnesses)
            parent2 = selection(population, fitnesses)
            # create two children using crossover
            child1, child2 = order_crossover(parent1, parent2)
            # create mutations in the children
            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)
            new_population.extend([child1, child2])
        
        # the old pop is irrelevant
        population = new_population
        # the best array is the one that has the minimal number of blocking pairs
        current_best_fitness = min(fitnesses)
        # extract the actual array
        current_best_solution = population[fitnesses.index(current_best_fitness)]
        
        # Update the best solution found (id the blocking pairs number is smaller than the current number of bp)
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = current_best_solution
        
        # Early stopping if a perfect solution is found
        if best_fitness == 0:
            break
    
    return best_solution, best_fitness

# Selection: tournament selection
def selection(population, fitnesses, k=3):
    # Randomly select k individuals and return the best one
    selected = random.choices(list(range(len(population))), k=k)
    selected_fitnesses = [fitnesses[i] for i in selected]
    return population[selected[np.argmin(selected_fitnesses)]]

# main
# Initialize preferences
file_path = r"C:\Users\User\Documents\ביולגיה חישובית\GA_input.txt"
men_prefs, women_prefs = read_prefs(file_path)
print("Men's Preferences:\n")
print(men_prefs)
print("Women's Preferences:\n")
print(women_prefs)
# Run the genetic algorithm
best_solution, best_fitness = genetic_algorithm()
print(f"Best Solution: {best_solution}")
print(f"Best Fitness: {best_fitness}")