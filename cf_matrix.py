
import html
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.metrics import confusion_matrix
import pandas as pd

def generate_plotly_cf_mat(y_true,y_pred, label_map ,figure_name:str,figure_path:str):
    labels=list(label_map.values())
    labels.sort()
    cf_matrix=confusion_matrix(y_true,y_pred,normalize="true")
    cf_diag=np.diag(cf_matrix)
    idx=np.argsort(cf_diag)
    cf_matrix=cf_matrix[idx,:][:,idx]
    cf_matrix=pd.DataFrame(cf_matrix)
    cat_names=[labels[i] for i in idx]
    fig = px.imshow(np.array(cf_matrix),
                labels=dict(x="Actual", y="Predicted"),
                x=cat_names,
                y=cat_names
               )
    fig.write_html(figure_path+figure_name, auto_open=True)
    fig.show()

if __name__=="__main__":
    label_map = ["CONTENT","TRIGGER","ACTION"]
    y_true=['TRIGGER', 'CONTENT', 'ACTION', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'TRIGGER', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT']
    y_pred=['CONTENT', 'TRIGGER', 'ACTION', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'TRIGGER', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT', 'CONTENT']
    generate_plotly_cf_mat(y_true,y_pred,label_map,"./","test.html")