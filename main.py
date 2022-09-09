import os

from datetime import datetime
from dobert import train_bert_based_models, grid_bert_based_models, test_bert_based_models, prediction_bert_based_models
from doxlnet import train_xlnet_model, grid_xlnet_model, test_xlnet_model
from create_doccano_json import createDoccannoJSON
from bs4 import BeautifulSoup
import html2text
from random import choice
from string import ascii_lowercase

ROOT_DIR = os.path.dirname(os.path.abspath('data.json'))

def train(modelType, modelPath, useCuda, nIterations):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start time = ", current_time)

    if(modelType in ["bert-base-cased", "allenai/scibert_scivocab_cased", "roberta-base", "xlnet-base-cased"]):
        train_bert_based_models(modelType, modelPath, useCuda, nIterations)
    else:
        train_xlnet_model(modelPath, useCuda, nIterations)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("End time = ", current_time)

def grid_search(modelType, modelPath, useCuda, nIterations):
    if (modelType in ["1", "2", "3", "4"]):
        grid_bert_based_models(modelType, modelPath, useCuda, nIterations)
    else:
        grid_xlnet_model(modelPath, useCuda, nIterations)

def test_model_dataset(modelType):
    if (modelType in ["1", "2", "3", "4"]):
        test_bert_based_models(modelType, modelPath)
    else:
        test_xlnet_model(modelType, modelPath)


if __name__ == '__main__':
    actionType = input("Action type: (1. Train; 2.Grid search; 3. Dataset Test; 4. "
                        "Manual Test; ; 5. Manual html file Test): ")
    if actionType == "":
        actionType = "3"

    modelTypeID = input("1. bert-base-cased; 2. scibert_scivocab_cased; 3. roberta-base; 4. distilbert-base-cased; 5. xlnet-base-cased: ")
    if modelTypeID == "1":
        modelType = "bert-base-cased"
    elif modelTypeID == "2":
        modelType = "allenai/scibert_scivocab_cased"
    elif modelTypeID == "3":
        modelType = "roberta-base"
    elif modelTypeID == "4":
        modelType = "distilbert-base-cased"
    elif modelTypeID == "5":
        modelType = "xlnet-base-cased"
    else:
        raise ValueError("The model ID: " + modelTypeID + "you selected does not exist")

    modelName = input("Enter the Model name: ")
    if modelName == "":
        modelName = ''.join(choice(ascii_lowercase) for i in range(12))

    modelPath = os.path.dirname(os.path.abspath(__file__)) + '/trained_models/' + modelName

    useCuda = input("Use Cuda? (y or n, default n): ")
    if useCuda == "y":
        useCuda = True
    else:
        useCuda = False

    if actionType == str(1) or actionType == str(2):
        nIterations = input("Number of iterations (default 10): ")
        if nIterations == "":
            nIterations = 10
        else:
            nIterations = int(nIterations)

        if actionType == str(1):
            train(modelType, modelPath, useCuda, nIterations)
        else:
            grid_search(modelType, modelPath, useCuda, nIterations)

    elif actionType == str(3):
        test_model_dataset(modelPath)

    elif actionType == str(4):
        text = input("Enter your testing text: ")
        if text == "":
            text = "The authors should correct the typos in the conclusion, those are visible within the conclusion in the lines: 22-28."
        predictionResults = prediction_bert_based_models(text, modelName)
        createDoccannoJSON(text, predictionResults)

    elif actionType == str(5):
        file_path = input("HTML file path with review comments: ")
        if file_path == "":
            file_path = "/Users/miloscuculovic/Desktop/eval2.html"

        file_object = open('doccano.json', 'a')
        with open(file_path, 'r') as f:
            contents = f.read()
            soup = BeautifulSoup(contents, 'lxml')
            paragraphs = soup.findAll('p')

            for paragraph in paragraphs:
                if paragraph.string is not None:
                    html2text.html2text.BODY_WIDTH = 0
                    text = html2text.html2text(paragraph.string)
                    if len(text) < 10:
                        continue
                    text = text.replace("\n", " ")
                    predictionResults = prediction_bert_based_models(text, modelName)

                    doccano = createDoccannoJSON(text, predictionResults)
                    if doccano != False:
                        file_object.write(doccano)

        file_object.close()
