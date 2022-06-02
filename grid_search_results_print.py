from pathlib import Path
import matplotlib.pyplot as plt
import numpy
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap


def generate_grid_search_results_print(grid_results, output_dir, model_name):
    weightdecay = []
    learningrate = []
    trainbatchsize = []
    f1 = []
    top_f1 = 0
    for key in grid_results:
        weightdecay.append(float(np.log10(grid_results[key][0])))
        learningrate.append(round(float(np.log10(grid_results[key][1])),2))
        trainbatchsize.append(float(grid_results[key][2]))
        f1.append(float(grid_results[key][3]))
        if top_f1 < f1[-1]:
            top_score = [weightdecay[-1], learningrate[-1], trainbatchsize[-1], f1[-1]]
            top_f1 = f1[-1]


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Grid search results - " + model_name)
    ax.set_xlabel("Log10(Wight decay)")
    ax.set_ylabel("Log10(Learning rate)")
    ax.set_zlabel("Batch size")
    ax.set_xticks(weightdecay)
    ax.set_yticks(learningrate)
    ax.set_zticks(trainbatchsize)

    arr = numpy.array(f1)
    new_col = arr.copy()

    NC = 5
    new_col[arr < 0.5] = 0
    new_col[(arr >= 0.5) & (arr < 0.75)] = 1
    new_col[(arr >= 0.75) & (arr < 0.8)] = 2
    new_col[(arr >= 0.8) & (arr < 0.85)] = 3
    new_col[arr >= 0.85] = 4
    #new_col[(arr >= 0.5) & (arr < 0.55)] = 1
    #new_col[(arr >= 0.55) & (arr < 0.60)] = 2
    #new_col[(arr >= 0.60) & (arr < 0.65)] = 3
    #new_col[(arr >= 0.65) & (arr < 1)] = 4
    new_col = new_col / NC




    cmap = ListedColormap(["magenta", "green", "blue", "orange", "red"])
    scat_plot = ax.scatter(xs=weightdecay, ys=learningrate, zs=trainbatchsize, c=cmap(new_col))
    ax.text(top_score[0], top_score[1], top_score[2], top_score[3], color="black")

    cb = fig.colorbar(cm.ScalarMappable(cmap=cmap), pad=0.2)
    cb.ax.set_xlabel('F1 score')
    cb.ax.set_yticklabels([0, 0.5, 0.75, 0.80, 0.85, 1])
    cb.ax.set_yticks(np.linspace(0, 1, 5 + 1), [0, 0.5, 0.75, 0.80, 0.85, 1])
    #cb.ax.set_yticklabels([0, 0.5, 0.55, 0.60, 0.65, 1])

    plt.plot(top_score[0], top_score[1], top_score[2], marker="o", markersize=15, markerfacecolor="yellow")

    path = Path(output_dir)
    plt.savefig(str(path.absolute()) + '/grid_search_plot_' + model_name + ".pdf")
    plt.show()

    print(top_score)
