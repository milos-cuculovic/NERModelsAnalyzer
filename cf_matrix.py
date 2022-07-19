import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.multiclass import unique_labels

def generate_confusion_matrix(y_true,y_pred, figure_path:str, model_name):
    map={"O":"O", "B-LOCATION":"LOCATION", "I-LOCATION":"LOCATION", "B-TRIGGER":"TRIGGER", "I-TRIGGER":"TRIGGER",
                  "B-MODAL":"MODAL", "I-MODAL":"MODAL", "B-ACTION":"ACTION", "I-ACTION":"ACTION"}
    y_true=[map[i] for i in y_true]

    y_pred=[map[i] for i in y_pred]

    labels=list(set(list(map.values())))
    labels.sort()

    cf_matrix=confusion_matrix(y_true,y_pred,normalize="true")
    cf_diag=np.diag(cf_matrix)
    idx=np.argsort(cf_diag)
    cat_names=[labels[i] for i in idx]
    cat_names.sort()

    plot_confusion_matrix(y_true, y_pred, cat_names, figure_path, model_name)


def plot_confusion_matrix(y_true, y_pred, classes,
                          figure_path, model_name):

    cf_matrix = confusion_matrix(y_true, y_pred, normalize="true")

    group_counts = ["{:.2f}".format(value) for value in
                    cf_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n" for v1, v2 in
              zip(group_counts, group_percentages)]

    labels = np.asarray(labels).reshape(5, 5)

    plt.figure(figsize = (9,9))
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues', square=True, cbar_kws={'label': 'F1 score'})
    ax.set_title("\n\n" + model_name + "\n\n");
    ax.set_xlabel('\nPredicted', fontsize=16)
    ax.set_ylabel('Actual\n', fontsize=16);
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)

    plt.savefig(figure_path + "/confusion_matrix.png")




if __name__=="__main__":
    label_map = ["CONTENT","TRIGGER","ACTION"]
    y_true=['TRIGGER', 'CONTENT', 'ACTION', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'TRIGGER', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT']
    y_pred=['CONTENT', 'TRIGGER', 'ACTION', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'TRIGGER', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT']
    generate_plotly_cf_mat(y_true,y_pred,label_map,"./","test.html")