from bert import Ner
model = Ner("out_ner/")

def prediction(text):
    output = model.predict(text)
    for prediction in output:
        print(prediction)

prediction("the review should be more explained")
