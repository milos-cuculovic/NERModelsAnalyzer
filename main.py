import os

from dospacy import trainSpacyModel
from dospacy import testSpacyModel
from dobilstm import trainBiLSTMModel
from datetime import datetime


def train_model(model, modelFile):
    LABEL = ['LOCATION', 'CONTENT', 'TRIGGER', 'MODAL', 'ACTION']
    LABEL = ['LOCATION', 'TRIGGER', 'MODAL', 'ACTION']

    ROOT_DIR = os.path.dirname(os.path.abspath('data.json'))
    path_csv = os.path.join(ROOT_DIR, 'selena2.csv')

    dropout = 1e-4
    nIter   = 100

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start time = ", current_time)

    if model == str(1):
        nlp = trainSpacyModel(path_csv, LABEL, dropout, nIter)
    else:
        if model == str(2):
            nlp = doBiLSTMModel(path_csv, LABEL, dropout, nIter)
        else:
            exit("Wrong model selection")

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("End time = ", current_time)

    # Save the trained Model
    nlp.to_disk(modelFile)


def test_model(model_type):
    model_name = input("Model name: ")
    number_of_testing_examples = input("Enter the number of testing examples: ")
    testSpacyModel(model_name, number_of_testing_examples)


if __name__ == '__main__':
    model_type = input("Model (1. spaCy; 2. Bi-LSTM): ")
    modelFile = input("Enter the Model name to save: ")
    train_model(model_type, modelFile)
    #test_model(model_type)
