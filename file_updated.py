import sys
import random
import numpy as np
import matplotlib.pyplot as plt

def read_prefs(file_path):
    # Read all lines from the input file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Calculate total number of lines and midpoint
    total_lines = len(lines)
    midpoint = total_lines // 2
    
    # Initialize preferences for men and women
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

def fitness(solution, men_prefs, women_prefs, total_lines):
    women_prefs_list = women_prefs.tolist()
    blocking_pairs = 0
    n = len(solution)
    
    # Iterate over each man in the solution
    for man in range(1, n-1):
        # Array of preferences of the current man
        current_man_pref = men_prefs[man-1]
        current_man_pref_list = current_man_pref.tolist()
        
        # Extract his wife
        woman = solution[man-1]
        
        # Extract his wife's score
        woman_index = current_man_pref_list.index(woman)
        
        # Go over the women he prefers upon his wife
        for index_other_woman in range(0, woman_index):
            # Extract the actual other woman
            other_woman = current_man_pref[index_other_woman]
            other_woman_pref = women_prefs_list[other_woman-1]
            
            # Check if the other woman's man index is greater
            man_index = other_woman_pref.index(man)
            other_womans_man = solution.index(other_woman) + 1
            index_other_womans_man = other_woman_pref.index(other_womans_man)
            
            if index_other_womans_man > man_index:
                blocking_pairs += 1
                break 
    
    happy_couples = total_lines // 2 - blocking_pairs
    return happy_couples

def order_crossover(parent1, parent2):
    # Perform order crossover on two parent solutions
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child1 = [-1] * size
    child2 = [-1] * size
    child1[start:end] = parent1[start:end]
    child2[start:end] = parent2[start:end]

    # Function to fill remaining positions in child
    def fill_remaining_positions(child, parent):
        pos = end
        for i in range(size):
            if parent[(end + i) % size] not in child:
                child[pos % size] = parent[(end + i) % size]
                pos += 1

    fill_remaining_positions(child1, parent2)
    fill_remaining_positions(child2, parent1)
    
    return child1, child2

def mutation(solution):
    # Perform mutation with a given probability
    a, b = random.sample(range(len(solution)), 2)
    solution[a], solution[b] = solution[b], solution[a]
    return solution

def selection(population, fitnesses, crowding_distances, k=3):
    # Randomly select k individuals and return the best one considering both fitness and crowding distance
    selected = random.choices(list(range(len(population))), k=k)
    selected_fitnesses = [fitnesses[i] for i in selected]
    best_index = selected[np.argmax(selected_fitnesses)]
   
    # In case of ties in fitness, use crowding distance to decide
    best_fitness = fitnesses[best_index]
    candidates = [i for i in selected if fitnesses[i] == best_fitness]
   
    if len(candidates) > 1:
        best_index = max(candidates, key=lambda i: crowding_distances[i])
   
    return population[best_index]

def calculate_crowding_distance(fitnesses):
    # Calculate crowding distance for diversity preservation
    crowding_distances = [0] * len(fitnesses)
    sorted_indices = np.argsort(fitnesses)
   
    # Set the boundary points' crowding distance to a large number
    crowding_distances[sorted_indices[0]] = float('inf')
    crowding_distances[sorted_indices[-1]] = float('inf')
   
    # Calculate crowding distance for each individual
    for i in range(1, len(fitnesses) - 1):
        crowding_distances[sorted_indices[i]] += (fitnesses[sorted_indices[i+1]] - fitnesses[sorted_indices[i-1]])
   
    return crowding_distances

def genetic_algorithm(men_prefs, women_prefs, total_lines, pop_size=20, max_generations=900):
    base_sequence = list(range(1, (total_lines // 2) + 1))
        
    # Initialize the population with shuffled solutions
    population = [random.sample(base_sequence, len(base_sequence)) for _ in range(pop_size)]
    best_solution = None
    best_fitness = 0
    
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
            # choose randomly
            toss = random.choices([0, 1], weights=[0.8, 0.2])[0]
            # choose the best fitness 
            if toss == 0:
                parent1 = selection(population, fitnesses, crowding_distance)
                parent2 = selection(population, fitnesses, crowding_distance)
                child1 = parent1
                child2 = parent2
                new_population.extend([child1, child2])
            else:
                # choose two parents randomly and create mutations and crossover
                parent1 = random.choice(population)
                parent2 = random.choice(population)
                child1, child2 = order_crossover(parent1, parent2)
                child1 = mutation(child1)
                child2 = mutation(child2)
                new_population.extend([child1, child2])
       
        population = new_population
        current_best_fitness = max(fitnesses)
        current_worst_fitness = min(fitnesses)
        current_average_fitness = np.mean(fitnesses)

        current_best_solution = population[fitnesses.index(current_best_fitness)]
        
        # Record fitness values for the current generation
        best_fitness_over_gens.append(current_best_fitness)
        worst_fitness_over_gens.append(current_worst_fitness)
        average_fitness_over_gens.append(current_average_fitness)
       
        # Update the best solution found
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = current_best_solution
         
       
        # Early stopping if a perfect solution is found
        if best_fitness == total_lines // 2:
            break
   
    return best_solution, best_fitness, best_fitness_over_gens, worst_fitness_over_gens, average_fitness_over_gens

if __name__ == "__main__":
    # Check if file_path argument is provided
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    men_prefs, women_prefs, total_lines = read_prefs(file_path)
    
    best_solution, best_fitness, best_fitness_over_gens, worst_fitness_over_gens, average_fitness_over_gens = genetic_algorithm(men_prefs, women_prefs, total_lines)
    print(f"Best Solution: {best_solution}")
    print(f"Best Fitness: {best_fitness}")
