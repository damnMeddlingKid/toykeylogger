from acoustic_model import posterior_frequency
from letter_frequencies import  letter_probability


def cost_func(symbol, observation, posterior):
    amplitude_likelihood = posterior[symbol].pdf(observation)
    return amplitude_likelihood


def sequence_cost(sequence, observations, keyboard):
    cost = 1
    posterior = posterior_frequency(keyboard)

    for letter, observation in zip(sequence, observations):
        cost *= posterior[letter].pdf(observation)

    return cost


def optimal_sequence(observations, keyboard):
    alphabet = letter_probability.keys()
    posterior = posterior_frequency(keyboard)
    message = []
    total_cost = 1

    for observation in observations:
        max_cost = None
        for letter in alphabet:
            cost = cost_func(letter, observation, posterior)
            if max_cost is None or cost > max_cost[0]:
                max_cost = (cost, letter)
        message.append(max_cost[1])
        total_cost *= max_cost[0]

    return ''.join(message), total_cost
