from hopfieldnetwork import HopfieldNetwork
from inputdatabuilder import InputDataBuilder

import numpy as np
import matplotlib.pyplot as plt


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

def main():
    show_noise = False
    verbose = False
    num_inputs = 100
    hn = HopfieldNetwork(num_inputs)
    
    plot_x = []
    plot_y = []
    count = 0

    # different noise levels
    for noise in np.arange(0, 1, 0.005):
        # train on a random array
        input_vec = create_random_array(num_inputs, 0.5)
        hn.store_information(input_vec)
        for _ in range(0, 10):
            # create new noise vector
            noisy_input = create_noisy_array(input_vec, noise)

            # recall
            recovered_array = hn.recall(noisy_input, 0)

            # count number correct
            num_correct = 0
            for i in range(0, len(recovered_array)):
                if recovered_array[i] == input_vec[i]:
                    num_correct += 1 
            percentage = 100 * num_correct / len(recovered_array)

            # visualize difference in vectors
            if show_noise and count % 20 == 0:
                plt.figure()
                plt.plot(input_vec)
                plt.plot(noisy_input)
                plt.show()
            count += 1

            plot_x.append(noise)
            plot_y.append(percentage)

    # plot
    plt.plot(plot_x, plot_y, '.')
    plt.show()


if __name__ == "__main__":
    main()