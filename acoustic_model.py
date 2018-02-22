from __future__ import unicode_literals
from scipy.stats import norm
from numpy.random import normal
from math import sqrt

initial_amplitude = 10
key_observation_variance = 3
factor = 0.001


def observed_mean_amplitude(keyboard, microphone):
    mic_x, mic_y = microphone
    amplitude = {}

    for key, positions in keyboard.items():
        key_x, key_y = positions
        amplitude[key] = initial_amplitude / (sqrt((mic_x - key_x) ** 2 + (mic_y - key_y) ** 2) * factor) ** 2

    return amplitude


def posterior_amplitude(keyboard, microphone):
    mean_amplitudes = observed_mean_amplitude(keyboard, microphone)
    pdfs = {}
    for key, mean in mean_amplitudes.items():
        pdfs[key] = norm(loc=mean, scale=key_observation_variance).pdf
    return pdfs


def acoustic_model_observations(text, keyboard, microphone):
    amplitudes = observed_mean_amplitude(keyboard, microphone)
    observations = []

    for letter in text:
        sample = normal(amplitudes[letter], key_observation_variance)
        observations.append(sample)

    return observations
