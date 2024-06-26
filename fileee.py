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
    array_of_neg_ones = np.full(total_lines // 2, -1)
    # initialize the 0th preference to be -1
    men_prefs.append(array_of_neg_ones)
    women_prefs.append(array_of_neg_ones)
    # Split lines into men's and women's preferences
    for i in range(midpoint):
        prefs = list(map(int, lines[i].strip().split()))
        men_prefs.append(prefs)
    
    for i in range(midpoint, total_lines):
        prefs = list(map(int, lines[i].strip().split()))
        women_prefs.append(prefs)
    
    return np.array(men_prefs), np.array(women_prefs), total_lines

def fitness(solution, men_prefs, women_prefs, total_lines):
    women_prefs_array = np.array(women_prefs)
    blocking_pairs = 0
    n = len(solution)
    print(f'n is {n}')
    # Iterate over each man in the solution
    for man in range(1, n):
        # Get preferences of the current man
        current_man_pref = men_prefs[man]
        # array convertion
        current_man_pref_array = np.array(current_man_pref)
        # Get his wife (woman assigned to him in the solution)
        woman = solution[man]

        print("np.where(current_man_pref_array == woman)[0]")
        print(np.where(current_man_pref_array == woman)[0])
        print("np.where(current_man_pref_array == woman)[0][0]")
        print(np.where(current_man_pref_array == woman)[0][0])
        # Find the index of his wife in his preferences
        woman_index = np.where(current_man_pref_array == woman)[0][0]

        print(f'current man pref array')
        print(current_man_pref_array)        
        # Iterate over the women he prefers more than his current wife
        for index_other_woman in range(woman_index):
            # Extract the other woman he prefers more than his current wife
            other_woman = current_man_pref_array[index_other_woman]
            
            # Get the preference list of the other woman
            other_woman_prefs = women_prefs[other_woman]
            other_woman_prefs_array = np.array(other_woman_prefs)


            man_index_in_other_woman_prefs = np.where(other_woman_prefs_array == man)[0][0]
            
            # Find the man assigned to the other woman in the current solution
            other_womans_man = np.where(solution == other_woman)[0][0]
            # Find the index of the other woman's man in the other woman's preferences
            index_other_womans_man = np.where(women_prefs_array[other_woman] == other_womans_man)[0][0]
            
            # Check if the current pair (man, woman) forms a blocking pair
            if (man_index_in_other_woman_prefs < index_other_womans_man):
                blocking_pairs += 1
                break  # No need to check further once a blocking pair is found
    
    # Calculate number of "happy" couples
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

def mutation(solution, mutation_rate=0.1):
    # Perform mutation with a given probability
    if random.random() < mutation_rate:
        # Select 2 positions to swap
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

def genetic_algorithm(men_prefs, women_prefs, total_lines, pop_size=30, max_generations=600, mutation_rate=0.1):
    # Initialize the population with random solutions
    population = [[-1] + random.sample(range(1, total_lines // 2 + 1), total_lines // 2) for _ in range(pop_size)]
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
        
        # Record fitness values for the current generation
        best_fitness_over_gens.append(current_best_fitness)
        worst_fitness_over_gens.append(current_worst_fitness)
        average_fitness_over_gens.append(current_average_fitness)
       
        # Update the best solution found
        print(f'current best fitness {best_fitness}')
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = current_best_solution
            print(f"Generation {generation}: Found new best fitness = {best_fitness}")
       
        # Early stopping if a perfect solution is found
        if best_fitness == total_lines // 2:
            print(f"Generation {generation}: Found perfect solution, stopping early.")
            break
   
    return best_solution, best_fitness, best_fitness_over_gens, worst_fitness_over_gens, average_fitness_over_gens

# Main code to run the genetic algorithm
file_path = r"C:\Users\Admin\Downloads\GA_input.txt"
men_prefs, women_prefs, total_lines = read_prefs(file_path)
print("Men's Preferences:")
print(men_prefs)
print("Women's Preferences:")
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