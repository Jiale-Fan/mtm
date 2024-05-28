from matplotlib import pyplot as plt
import numpy as np

def plot_2D_rope(keypoints, path="./output/rope.png"):
    """
        keypoints: [33, ]
    """
    keypoints = keypoints.reshape(-1, 3)
    plt.plot(keypoints[:-1, 0], keypoints[:-1, 2], 'ro-') # plot x, z plane
    plt.savefig(path)
    plt.show()