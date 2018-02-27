import viterbi_algorithm
import ml_algorithm
import brute_force_algorithm
from acoustic_model import acoustic_model_observations
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


if __name__ == '__main__':

    keyboard = construct_keyboard(keys_per_row)

    num_samples = 1
    viterbi_average = 0
    ml_average = 0
    bf_average = 0

    for word in common_words_100:
        viterbi_probaility = 0
        ml_probability = 0
        bf_probability = 0

        for x in range(num_samples):
            observations = acoustic_model_observations(word, keyboard)
            viterbi_message = viterbi_algorithm.optimal_sequence(observations, keyboard)
            ml_message = ml_algorithm.optimal_sequence(observations, keyboard)
            bf_message = brute_force_algorithm.optimal_sequence(observations, keyboard)

            if viterbi_message[0] == word:
                viterbi_probaility += 1

            if ml_message[0] == word:
                ml_probability += 1

            if bf_message[0] == word:
                bf_probability += 1

            print("actual: {}, viterbi: {}, ml: {}, bf: {}".format(word, viterbi_message[0], ml_message[0], bf_message[0]))

        viterbi_average += viterbi_probaility / float(num_samples)
        ml_average += ml_probability / float(num_samples)
        bf_average += bf_probability / float(num_samples)

    print("map success: {} ml success: {} bf sucess: {}".format(viterbi_average/100.0, ml_average/100.0, bf_average/100.0))
