from scipy.stats import norm

key_observation_variance = 0.3
central_frequencies = range(13)
frequency_groups = [norm(loc=mean, scale=key_observation_variance) for mean in central_frequencies]


def posterior_frequency(keyboard):
    pdfs = {}
    letter = "qwertyuiopasdfghjklzxcvbnm"
    for index, key in enumerate(letter):
        idx = index % len(frequency_groups)
        pdfs[key] = frequency_groups[idx]
    return pdfs


def acoustic_model_observations(text, keyboard):
    observations = []
    key_frequencies = posterior_frequency(keyboard)

    for letter in text:
        sample = key_frequencies[letter].rvs(size=1, random_state=10)[0]
        observations.append(sample)

    return observations
