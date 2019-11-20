import numpy as np
import matplotlib.pyplot as plt


def plot_from_data(data):
    R = data[:1024].reshape(32,32)
    G = data[1024:2048].reshape(32,32)
    B = data[2048:3072].reshape(32,32)
    img = np.dstack([R,G,B])
    plt.imshow(img)
