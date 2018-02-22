import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from words import common_words_100
from acoustic_model import observed_mean_amplitude

keys_per_row = ["qwertyuiop[]", "asdfghjkl;'", "zxcvbnm,./", " "]
key_width = 5
key_spacing = 0.3


def construct_keyboard(keys_per_row):
    keyboard = {}

    for row_number, row_keys in enumerate(keys_per_row):
        x_start = -31 + (row_number * key_width * 0.5)
        y_start = -(row_number * (key_width + key_spacing))
        for index, key in enumerate(row_keys):
            key_y = y_start
            key_x = x_start + (index * (key_width + key_spacing))
            keyboard[key] = (key_x, key_y)

    return keyboard


def gaussian_intersections(keyboard, microphone):
    intersections = {}
    amplitudes = observed_mean_amplitude(keyboard, microphone)
    keys = keyboard.keys()
    keys = sorted(keys, key=lambda elem: amplitudes[elem])

    for index, key in enumerate(keys):
        key_distance = amplitudes[key]

        if index == 0:
            left_most = None
        else:
            nearest_left = amplitudes[keys[index - 1]]
            left_most = (key_distance ** 2 - nearest_left ** 2) / (2 * (key_distance - nearest_left))

        if index == len(keys) - 1:
            right_most = None
        else:
            nearest_right = amplitudes[keys[index + 1]]
            right_most = (key_distance ** 2 - nearest_right ** 2) / (2 * (key_distance - nearest_right))

        intersections[key] = (key_distance, left_most, right_most)

    return intersections


def probability_of_error_acoustic_model(keyboard, microphone):
    letter_success = {}
    average_error_probability = 0
    intersections = gaussian_intersections(keyboard, microphone)
    scaling_factor = 1/(2 * math.sqrt(2) * math.sqrt(key_observation_variance))

    for word in common_words_100:
        total_error_probability = 1

        for letter in word:
            letter = letter.lower()
            if letter not in letter_success:
                letter_distance, left_inter, right_inter = intersections[letter]

                if left_inter:
                    left_point = (left_inter - letter_distance) / (2 * scaling_factor)

                if right_inter:
                    right_point = (right_inter - letter_distance) / (2 * scaling_factor)

                if left_inter and right_inter:
                    success_probability = scaling_factor * (math.erf(right_point) - math.erf(left_point))
                elif right_inter and not left_inter:
                    success_probability = 0.5 + (scaling_factor * math.erf(right_point))
                elif left_inter and not right_inter:
                    success_probability = 0.5 + (-scaling_factor * math.erf(left_point))
                else:
                    success_probability = 1  # this should never happen

                letter_success[letter] = success_probability

            total_error_probability *= letter_success[letter]

        average_error_probability += math.log(total_error_probability)

    return average_error_probability / float(len(common_words_100))


if __name__ == '__main__':
    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111, aspect='equal')
    keyboard = construct_keyboard(keys_per_row)
    microphone = (40, 20)

    # left_extent = -100
    # right_extent = 100
    #
    # ax2.set_xlim(left_extent, right_extent)
    # ax2.set_ylim(left_extent, right_extent)
    #
    # circle = plt.Circle(microphone, 3, color='r')
    # #ax2.add_artist(circle)
    #
    # #plt.show()
    #
    # image = np.zeros((2 * right_extent, 2 * right_extent))
    #
    # for x in range(image.shape[1]):
    #     for y in range(image.shape[0]):
    #         x_coord = left_extent + x
    #         y_coord = right_extent - y
    #         value = 0.0
    #
    #         try:
    #             value = probability_of_error_acoustic_model(keyboard, (x_coord, y_coord))
    #         except Exception:
    #             value = -35.99344250211817
    #
    #         image[y, x] = value
    # print(image.max())
    # image *= 255 - (255.0 * image.max())
    # ax2.imshow(image, extent=[left_extent, right_extent, left_extent, right_extent])
    #
    #
    # for key, position in keyboard.items():
    #     x, y = position
    #
    #     ax2.add_patch(
    #         patches.Rectangle(
    #             (x, y),
    #             key_width if key != " " else 9.5 * key_width,
    #             key_width,
    #             edgecolor="black",
    #             facecolor="white"
    #         )
    #     )
    #     ax2.text(x + 1.2, y + 1, key)
    #
    # plt.show()

    from acoustic_model import acoustic_model_observations, posterior_amplitude, observed_mean_amplitude
    from viterbi import optimal_sequence, sequence_cost, sequence_cost_naive, optimal_sequence_naive

    num_samples = 1
    average_map = 0
    average_naive = 0

    for word in common_words_100:
        probaility_map = 0
        probability_naive = 0
        for x in range(num_samples):
            sequence = "refrigerator"
            observations = acoustic_model_observations(sequence, keyboard, microphone)
            message = optimal_sequence(observations, keyboard, microphone)
            message_naive = optimal_sequence_naive(observations, keyboard, microphone)
            posterior = posterior_amplitude(keyboard, microphone)
            means = observed_mean_amplitude(keyboard, microphone)

            if message[0] == sequence:
                probaility_map += 1

            if message_naive[0] == sequence:
                probability_naive += 1

        average_map += probaility_map / float(num_samples)
        average_naive += probability_naive / float(num_samples)

    print("map success: {} ml success: {}".format(average_map/100.0, average_naive/100.0))
    # print(message)
    # print(message_naive)
    # print('map {}, ml {}'.format(sequence_cost(sequence, observations, keyboard, microphone), sequence_cost_naive(sequence, observations, keyboard, microphone)))
    # print('map {}, ml {}'.format(sequence_cost(message[0], observations, keyboard, microphone), sequence_cost_naive(message[0], observations, keyboard, microphone)))
