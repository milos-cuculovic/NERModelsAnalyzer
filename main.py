import os

from dospacy import trainSpacyModel
from dospacy import testSpacyModel
from dospacy import evaluateSpacy
from dobilstm import trainBiLSTMModel
from datetime import datetime
from dobert import trainBERTModel, evaluation, pip_aggregation, Ner, prediction,trainBERTGrid
from doroberta import trainROBERTAModel, evaluationRoberta, pip_aggregationRoberta, predictionRoberta,trainROBERTAGrid
from doxlnet import trainxlnetModel, trainxlnetGrid
from create_doccano_json import createDoccannoJSON

def train_model(model, output_dir, useCuda, spacy_model_type = "1", grid_type = "1"):
    LABEL = ['LOCATION', 'CONTENT', 'TRIGGER', 'MODAL', 'ACTION']

    ROOT_DIR = os.path.dirname(os.path.abspath('data.json'))
    path_train_data_bert = os.path.join(ROOT_DIR, 'train.json')
    path_train_data = os.path.join(ROOT_DIR, 'train_temp_test.spacy')
    path_valid_data = os.path.join(ROOT_DIR, 'valid_temp_test.spacy')

    dropout = 1e-5
    nIter   = 1

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start time = ", current_time)

    if model == str(1):
        nlp, plt = trainSpacyModel(path_train_data, path_valid_data, LABEL, dropout, nIter, spacy_model_type)
    elif model == str(2):
        nlp = trainBiLSTMModel(path_train_data, LABEL, dropout, nIter, output_dir)
    elif model == str(3):
        trainBERTModel(path_train_data_bert, output_dir, nIter, useCuda)
        exit()
    elif model==str(4):
        print(grid_type)
        if grid_type=="1":
            trainBERTGrid(path_train_data_bert, output_dir, nIter, useCuda)
        elif grid_type=="2":
            trainROBERTAGrid(path_train_data_bert, output_dir, nIter, useCuda)
        elif grid_type=="3":
            trainxlnetGrid(path_train_data_bert, output_dir, nIter, useCuda)
        exit()
    elif model==str(5):
        trainROBERTAModel(path_train_data_bert, output_dir, nIter, useCuda)
        exit()
    elif model == str(6):
        trainxlnetModel(path_train_data_bert, output_dir, nIter, useCuda)
        exit()

    else:
        exit("Wrong model selection")

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("End time = ", current_time)

    # Save the trained Model
    nlp.to_disk(output_dir)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(output_dir + '/losses_graph.png')


def test_model_manually(model_path):
    number_of_testing_examples = input("Enter the number of testing examples: ")
    testSpacyModel(model_path, number_of_testing_examples)

def test_model_dataset(model_name):
    choice = input("1: BERT; 2:SpaCy; 3:RoBerta : ")
    if choice == "1" or choice=="3":
        useCuda = input("Use Cuda? (y or n, default n): ")
        if useCuda == "y":
            useCuda = True
        else:
            useCuda = False
        if choice == "1":
            evaluation(model_name, useCuda)
        else:
            evaluationRoberta(model_name, useCuda)
    else:
        #model_path = os.path.dirname(os.path.abspath(__file__)) + '/trained_models/' + model_name
        #LABEL = ['LOCATION', 'CONTENT', 'TRIGGER', 'MODAL', 'ACTION']
        LABEL = ['LOCATION', 'TRIGGER', 'MODAL', 'ACTION']
        ROOT_DIR = os.path.dirname(os.path.abspath('data.json'))
        path_test_data = os.path.join(ROOT_DIR, 'valid_temp_test.spacy')
        results = evaluateSpacy(model_name, path_test_data, LABEL)
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
    useCuda = False
    spacy_model_type = "1"
    action_type = input("Action type: (1. Train; 2. Dataset Test; 3. Manual Test; 4.Grid search): ")

    if action_type == "":
        action_type = "3"

    if action_type == str(1):
        model = input("Model (1. spaCy; 2. Bi-LSTM; 3. BERT; 4. BERT-pip_aggregation, 5.Roberta, 6: xlnet): ")
        modelFile = input("Enter the Model name to save: ")
        if model == "4":
            pip_aggregation(modelFile, modelFile + "_pip_aggregation")
            exit()
        if model == "3" or model == "4" :
            useCuda = input("Use Cuda? (y or n, default n): ")
            if useCuda == "y":
                useCuda = True
            else:
                useCuda = False
        if model == "5" or model=="6":
            useCuda = input("Use Cuda? (y or n, default n): ")
            if useCuda == "y":
                useCuda = True
            else:
                useCuda = False
        if model == "1":
            spacy_model_type = input("1. Blank; 2. en_core_web_trf; 3. en_core_web_sm (Default 1): ")
        train_model(model, os.path.dirname(os.path.abspath(__file__)) + '/trained_models/' + modelFile, useCuda, spacy_model_type)
    elif action_type==str(4):
        grid_type = input("Grid type: (1. Bert; 2. RoBerta; 3. XLNet): ")
        modelFile = input("Enter the Model name to save: ")
        useCuda = input("Use Cuda? (y or n, default n): ")
        if useCuda == "y":
                useCuda = True
        else:
                useCuda = False
        train_model(action_type, os.path.dirname(os.path.abspath(__file__)) + '/trained_models/' + modelFile, useCuda, "1", grid_type)
    else:
        if action_type == str(2):
            model_name = input("Model name to test: ")
            model_path = os.path.dirname(os.path.abspath(__file__)) + '/trained_models/' + model_name
            test_model_dataset(model_path)
        else:
            if action_type == str(3):
                model = input("Model (1. spaCy; 2.BERT; 3.RoBerta): ")

                if model == "":
                    model = "2"

                if model == "1":
                    model_name = input("Model name to test: ")
                    model_path = os.path.dirname(os.path.abspath(__file__)) + '/trained_models/' + model_name
                    test_model_manually(model_path)
                elif model == "2":
                    model_name = input("Model name to test: ")

                    if model_name == "":
                        model_name = "scibert_grid_45_29.04.2022"

                    text = input("Enter your testing text: ")
                    if text == "":
                        text = "The authors should correct the typos in the conclusion, those are visible within the conclusion in the lines: 22-28."
                    predictionResults = prediction(text, model_name)
                    print(predictionResults)
                    createDoccannoJSON(text, predictionResults)


                elif model =="3":
                    model_name=input("Model name to test: ")
                    text = input("Enter your testing text: ")
                    print(predictionRoberta(text, model_name))
            else:
                train_model("1", os.path.dirname(os.path.abspath(__file__)) + '/trained_models/1_default', useCuda, spacy_model_type)