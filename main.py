from words import common_words_100

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


# def gaussian_intersections(keyboard, microphone):
#     intersections = {}
#     amplitudes = observed_mean_amplitude(keyboard, microphone)
#     keys = keyboard.keys()
#     keys = sorted(keys, key=lambda elem: amplitudes[elem])
#
#     for index, key in enumerate(keys):
#         key_distance = amplitudes[key]
#
#         if index == 0:
#             left_most = None
#         else:
#             nearest_left = amplitudes[keys[index - 1]]
#             left_most = (key_distance ** 2 - nearest_left ** 2) / (2 * (key_distance - nearest_left))
#
#         if index == len(keys) - 1:
#             right_most = None
#         else:
#             nearest_right = amplitudes[keys[index + 1]]
#             right_most = (key_distance ** 2 - nearest_right ** 2) / (2 * (key_distance - nearest_right))
#
#         intersections[key] = (key_distance, left_most, right_most)
#
#     return intersections
#
#
# def probability_of_error_acoustic_model(keyboard, microphone):
#     letter_success = {}
#     average_error_probability = 0
#     intersections = gaussian_intersections(keyboard, microphone)
#     scaling_factor = 1/(2 * math.sqrt(2) * math.sqrt(key_observation_variance))
#
#     for word in common_words_100:
#         total_error_probability = 1
#
#         for letter in word:
#             letter = letter.lower()
#             if letter not in letter_success:
#                 letter_distance, left_inter, right_inter = intersections[letter]
#
#                 if left_inter:
#                     left_point = (left_inter - letter_distance) / (2 * scaling_factor)
#
#                 if right_inter:
#                     right_point = (right_inter - letter_distance) / (2 * scaling_factor)
#
#                 if left_inter and right_inter:
#                     success_probability = scaling_factor * (math.erf(right_point) - math.erf(left_point))
#                 elif right_inter and not left_inter:
#                     success_probability = 0.5 + (scaling_factor * math.erf(right_point))
#                 elif left_inter and not right_inter:
#                     success_probability = 0.5 + (-scaling_factor * math.erf(left_point))
#                 else:
#                     success_probability = 1  # this should never happen
#
#                 letter_success[letter] = success_probability
#
#             total_error_probability *= letter_success[letter]
#
#         average_error_probability += math.log(total_error_probability)
#
#     return average_error_probability / float(len(common_words_100))


if __name__ == '__main__':
    import viterbi_algorithm
    import ml_algorithm
    from acoustic_model import acoustic_model_observations
    keyboard = construct_keyboard(keys_per_row)

    num_samples = 1
    viterbi_average = 0
    ml_average = 0

    for word in common_words_100:
        viterbi_probaility = 0
        ml_probability = 0
        for x in range(num_samples):
            observations = acoustic_model_observations(word, keyboard)
            viterbi_message = viterbi_algorithm.optimal_sequence(observations, keyboard)
            ml_message = ml_algorithm.optimal_sequence(observations, keyboard)

            # print("actual {}".format(word))
            # print("Viterbi estimate {} score {}".format(viterbi_message[0], viterbi_message[1]))
            # print("ML estimate {} score {}".format(ml_message[0], ml_message[1]))

            if viterbi_message[0] == word:
                viterbi_probaility += 1

            if ml_message[0] == word:
                ml_probability += 1

        viterbi_average += viterbi_probaility / float(num_samples)
        ml_average += ml_probability / float(num_samples)

    print("map success: {} ml success: {}".format(viterbi_average/100.0, ml_average/100.0))
