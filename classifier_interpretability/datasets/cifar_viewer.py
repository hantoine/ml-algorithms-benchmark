import numpy as np
import matplotlib.pyplot as plt


def plot_from_data(data):
    img = np.moveaxis(data.reshape(3, 32, 32), [0, 1, 2], [2, 0, 1])
    plt.imshow(img)
