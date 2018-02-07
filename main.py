import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numpy.random import normal

keys_per_row = ["qwertyuiop[]", "asdfghjkl;'", "zxcvbnm,./", " "]
key_width = 5
key_spacing = 0.3
key_observation_variance = 0.5


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


def squared_distance_to_keys(keyboard, microphone):
    mic_x, mic_y = microphone
    distance = {}

    for key, positions in keyboard.items():
        key_x, key_y = positions
        distance[key] = (mic_x - key_x) ** 2 + (mic_y - key_y) ** 2

    return distance


def acoustic_model(text, keyboard, microphone):
    distance = squared_distance_to_keys(keyboard, microphone)
    observations = []

    for letter in text:
        sample = normal(distance[letter], key_observation_variance)
        observations.append(sample)

    return observations


if __name__ == '__main__':
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, aspect='equal')
    keyboard = construct_keyboard(keys_per_row)

    ax2.set_xlim(-50, 50)
    ax2.set_ylim(-50, 50)

    for key, position in keyboard.items():
        x, y = position

        ax2.add_patch(
            patches.Rectangle(
                (x, y),
                key_width if key != " " else 9.5 * key_width,
                key_width,
                fill=False
            )
        )
        ax2.text(x + 1.2, y + 1, key)

    plt.show()
