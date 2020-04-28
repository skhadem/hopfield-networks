import numpy as np


def create_random_array(length, percentage_positive):
    random_array = []

    for i in range(0, length):
        if np.random.rand() > percentage_positive:
            random_array.append(1)
        else:
            random_array.append(-1)

    return np.array(random_array)

def create_noisy_array(input_array, percent_to_flip):
    noisy_input = np.copy(input_array)
    
    for i in range(0, len(noisy_input)):
        if np.random.rand() > (1 - percent_to_flip):
            # flip bit
            noisy_input[i] = -noisy_input[i]

    return np.array(noisy_input)
