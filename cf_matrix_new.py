import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):

    cf_matrix = confusion_matrix(y_true, y_pred, normalize="true")

    #ax = sns.heatmap(cf_matrix, annot=True, cmap=cmap)
    #ax.set_title(title);
    #ax.set_xlabel('\nPredicted Values')
    #ax.set_ylabel('Actual Values ');
    #ax.xaxis.set_ticklabels(classes)
    #ax.yaxis.set_ticklabels(classes)

    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n" for v1, v2 in
              zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(5, 5)
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Flower Category')
    ax.set_ylabel('Actual Flower Category ');
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)

    plt.savefig("fig1.png")