from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path


def generate_grid_search_results_print(grid_results, output_dir):
    weightdecay = []
    learningrate = []
    trainbatchsize = []
    f1 = []

    for key in grid_results:
        weightdecay.append(float(np.log10(grid_results[key][0])))
        learningrate.append(float(grid_results[key][1]))
        trainbatchsize.append(float(grid_results[key][2]))
        f1.append(float(grid_results[key][3]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xticks(weightdecay)
    ax.set_yticks(learningrate)
    img = ax.scatter(weightdecay, learningrate, trainbatchsize, c=f1, cmap="Paired")
    fig.colorbar(img)

    path = Path(output_dir)
    plt.savefig(str(path.parent.absolute()) + '/grid_search_plot.pdf')