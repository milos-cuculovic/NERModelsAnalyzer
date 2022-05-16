from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

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

    scat_plot = ax.scatter(xs=weightdecay, ys=learningrate, zs=trainbatchsize, c=f1, cmap="bwr")
    ax.text(top_score[0], top_score[1], top_score[2], top_score[3], color="black")

    cb = plt.colorbar(scat_plot, pad=0.2)
    cb.ax.set_xlabel('F1 score')

    plt.plot(top_score[0], top_score[1], top_score[2], marker="o", markersize=15, markerfacecolor="yellow")

    path = Path(output_dir)
    plt.savefig(str(path.parent.absolute()) + '/grid_search_plot_' + model_name + ".pdf")
    plt.show()

