from __future__ import unicode_literals
from letter_frequencies import bigram_probability, letter_probability
from acoustic_model import posterior_amplitude, acoustic_model_observations


def cost_func(symbol, previous_symbol, observation, posterior):
    amplitude_likelihood = posterior[symbol](observation)

    if previous_symbol is None:
        amplitude_likelihood *= letter_probability[symbol]
    else:
        letters = letter_probability.keys()
        sum = 0
        for letter in letters:
            sum += bigram_probability[previous_symbol + letter]
        amplitude_likelihood *= (bigram_probability[previous_symbol + symbol] / sum)

    return amplitude_likelihood


def cost_func_naive(symbol, previous_symbol, observation, posterior):
    amplitude_likelihood = posterior[symbol](observation)
    return amplitude_likelihood


def sequence_cost(sequence, observations, keyboard, microphone):
    cost = 1
    previous_symbol = None
    posterior = posterior_amplitude(keyboard, microphone)

    for letter, observation in zip(sequence, observations):
        cost *= cost_func(letter, previous_symbol, observation, posterior)

    return cost

def sequence_cost_naive(sequence, observations, keyboard, microphone):
    cost = 1
    previous_symbol = None
    posterior = posterior_amplitude(keyboard, microphone)

    for letter, observation in zip(sequence, observations):
        cost *= posterior[letter](observation)

    return cost


def recursive_optimal_sequence(posterior, observations, alphabet, index, index_symbol_cost, previous_symbol, naive=False):
    max_cost = None

    for symbol in alphabet:
        sequence_key = (symbol, index)
        cost = cost_func(symbol, previous_symbol, observations[index], posterior)

        if naive:
            cost = cost_func_naive(symbol, previous_symbol, observations[index], posterior)

        if sequence_key not in index_symbol_cost:
            if index != len(observations) - 1:
                max_residual = recursive_optimal_sequence(posterior, observations, alphabet, index + 1, index_symbol_cost, symbol)
                cost *= max_residual[0]
                index_symbol_cost[sequence_key] = max_residual
        else:
            cost *= index_symbol_cost[sequence_key][0]

        if max_cost is None or cost > max_cost[0]:
            max_cost = (cost, symbol)

    return max_cost


def optimal_sequence(amplitudes, keyboard, microphone):
    alphabet = letter_probability.keys()
    sequence_length = len(amplitudes)
    cost_table = dict()
    posterior = posterior_amplitude(keyboard, microphone)

    best_start = recursive_optimal_sequence(posterior, amplitudes, alphabet, 0, cost_table, None)
    message = [best_start[1]]
    key = (best_start[1], 0)

    for index in range(1, len(amplitudes)):
        cost = cost_table[key]
        message.append(cost[1])
        key = (cost[1], index) # this might error out on the index

    return ''.join(message), best_start[0]


def optimal_sequence_naive(amplitudes, keyboard, microphone):
    alphabet = letter_probability.keys()
    sequence_length = len(amplitudes)
    cost_table = dict()
    posterior = posterior_amplitude(keyboard, microphone)

    best_start = recursive_optimal_sequence(posterior, amplitudes, alphabet, 0, cost_table, None, naive=True)
    message = [best_start[1]]
    key = (best_start[1], 0)

    for index in range(1, len(amplitudes)):
        cost = cost_table[key]
        message.append(cost[1])
        key = (cost[1], index) # this might error out on the index

    return ''.join(message), best_start[0]
