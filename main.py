import os

from dospacy import trainSpacyModel
from dospacy import testSpacyModel
from dospacy import evaluateSpacy
from dobilstm import trainBiLSTMModel
from datetime import datetime
from dobert import trainBERT, evaluation

def train_model(model, modelFile):
    LABEL = ['LOCATION', 'CONTENT', 'TRIGGER', 'MODAL', 'ACTION']
    #LABEL = ['LOCATION', 'TRIGGER', 'MODAL', 'ACTION']

    ROOT_DIR = os.path.dirname(os.path.abspath('data.json'))
    path_train_data = os.path.join(ROOT_DIR, 'data_train_small.json')

    dropout = 1e-5
    nIter   = 100

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start time = ", current_time)

    if model == str(1):
        nlp = trainSpacyModel(path_train_data, LABEL, dropout, nIter)
    elif model == str(3):
        nlp = trainBERT(ROOT_DIR + "/" + modelFile + ".json")
        exit()
    else:
        if model == str(2):
            nlp = trainBiLSTMModel(path_train_data, LABEL, dropout, nIter)
        else:
            exit("Wrong model selection")

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("End time = ", current_time)

    # Save the trained Model
    nlp.to_disk(os.path.dirname(os.path.abspath(__file__)) + '/trained_models/' + modelFile)


def test_model_manually(model_path):
    number_of_testing_examples = input("Enter the number of testing examples: ")
    testSpacyModel(model_path, number_of_testing_examples)

def test_model_dataset(model_path,modelname):
    choice=input("1:berteval 0:else")
    if choice=="1":
        evaluation(data_name,model_name)
    else:
        model_path= os.path.dirname(os.path.abspath(__file__)) + '/trained_models/' + model_name
        LABEL = ['LOCATION', 'CONTENT', 'TRIGGER', 'MODAL', 'ACTION']
        ROOT_DIR = os.path.dirname(os.path.abspath('data.json'))
        path_test_data = os.path.join(ROOT_DIR, 'data_test_full.json')
        results = evaluateSpacy(model_path, path_test_data, LABEL)
        print("Entity\t\tPrecision\tRecall\tF-score")
        for result in results['ents_per_type']:
            if(result == "LOCATION"):
                print(
                    "{:}\t{:0.4f}\t{:0.4f}\t{:0.4f}".
                        format(result, results['ents_per_type'][result]['p'], results['ents_per_type'][result]['r'],
                               results['ents_per_type'][result]['f'])
                )
            else:
                print(
                    "{:}\t\t{:0.4f}\t{:0.4f}\t{:0.4f}".
                        format(result, results['ents_per_type'][result]['p'], results['ents_per_type'][result]['r'], results['ents_per_type'][result]['f'])
                )
        print()
        print(
            "{:}\t\t{:0.4f}\t{:0.4f}\t{:0.4f}".
                format("TOTAL", results['ents_p'], results['ents_r'], results['ents_f'])
        )


if __name__ == '__main__':
    action_type = input("Action type: (1. Train; 2. Dataset Test; 3. Manual Test;): ")
    if action_type == str(1):
        model = input("Model (1. spaCy; 2. Bi-LSTM; 3. BERT): ")
        modelFile = input("Enter the Model name to save: ")
        train_model(model, modelFile)
    else:
        if action_type == str(2):
            data_name = input("datafile name to test: ")
            model_name = input("Model name to test: ")
            test_model_dataset(data_name , model_name)

        else:
            model_name = input("Model name to test: ")
            model_path = os.path.dirname(os.path.abspath(__file__)) + '/trained_models/' + model_name
            test_model_manually(model_path)