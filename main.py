import os

from dospacy import trainSpacyModel
from dospacy import testSpacyModel
# from dobilstm import trainBiLSTMModel
from datetime import datetime
from dobert import trainBERT, evaluation


def train_model(model, modelFile):
    LABEL = ['LOCATION', 'CONTENT', 'TRIGGER', 'MODAL', 'ACTION']
    # LABEL = ['LOCATION', 'TRIGGER', 'MODAL', 'ACTION']

    ROOT_DIR = os.path.dirname(os.path.abspath('data.json'))
    path_csv = os.path.join(ROOT_DIR, 'data_full_test.csv')

    dropout = 1e-4
    nIter = 10

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start time = ", current_time)

    if model == str(1):
        nlp = trainSpacyModel(path_csv, LABEL, dropout, nIter)
        # Save the trained Model
        nlp.to_disk(os.path.dirname(os.path.abspath(__file__)) + '/trained_models/' + modelFile)
    elif model == str(3):
        nlp = trainBERT(ROOT_DIR+"/"+modelFile + ".json")
    else:
        if model == str(2):
            print("no")
            # nlp = trainBiLSTMModel(path_csv, LABEL, dropout, nIter)
            # Save the trained Model
            #nlp.to_disk(modelFile)
        else:
            exit("Wrong model selection")

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("End time = ", current_time)



def test_model_manually(model_name):
    number_of_testing_examples = input("Enter the number of testing examples: ")
    testSpacyModel(model_name, number_of_testing_examples)


def test_model(data_name, model_name):
    evaluation(data_name,model_name)


if __name__ == '__main__':
    action_type = input("Action type: (1. Train; 2. Dataset Test; 3. Manual Test;): ")
    if action_type == str(1):
        model_type = input("Model (1. spaCy; 2. Bi-LSTM; 3. BERT): ")
        modelFile = input("Enter the Model name to save: ")
        train_model(model_type, modelFile)
    else:
        if action_type == str(2):
            data_name = input("datafile name to test: ")
            model_name = input("modelfile name to test: ")
            # test_model_dataset(model_name)
            test_model( data_name , model_name)

        else:
            model_name = input("Model name to test: ")
            test_model_manually(model_name)