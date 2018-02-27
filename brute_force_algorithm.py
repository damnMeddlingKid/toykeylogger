import enchant
from viterbi_algorithm import cost_func
from letter_frequencies import letter_probability
from acoustic_model import posterior_frequency


dictionary = enchant.Dict("en_US")


def recursive_brute_force(alphabet, observations, posterior, previous_symbol, index, message='', previous_cost=1):
    max_cost = None

    if index == len(observations):
        if not dictionary.check(message):
            previous_cost = -float('inf')
        return message, previous_cost

    for symbol in alphabet:
        cost = previous_cost * cost_func(symbol, previous_symbol, observations[index], posterior)
        total_cost = recursive_brute_force(alphabet, observations, posterior, symbol, index + 1, message + symbol, cost)

        if max_cost is None or total_cost[1] > max_cost[1]:
            max_cost = total_cost

    return max_cost


def optimal_sequence(observations, keyboard):
    alphabet = letter_probability.keys()
    posterior = posterior_frequency(keyboard)

    message = recursive_brute_force(alphabet, observations, posterior, None, 0)

    return message