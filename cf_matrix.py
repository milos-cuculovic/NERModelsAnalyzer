
import html
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.metrics import confusion_matrix
import pandas as pd

def generate_plotly_cf_mat(y_true,y_pred, label_map ,figure_name:str,figure_path:str):
    map={"O":"O", "B-LOCATION":"LOCATION", "I-LOCATION":"LOCATION", "B-TRIGGER":"TRIGGER", "I-TRIGGER":"TRIGGER",
                  "B-MODAL":"MODAL", "I-MODAL":"MODAL", "B-ACTION":"ACTION", "I-ACTION":"ACTION", "B-CONTENT":"CONTENT", "I-CONTENT":"CONTENT"}
    y_true=[map[i] for i in y_true]
    print(y_true[:20])
    y_pred=[map[i] for i in y_pred]
    print(y_pred[:20])
    labels=list(set(list(map.values())))
    labels.sort()
    print(labels)
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