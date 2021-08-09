import bert
from bert import Ner

model = Ner("out_ner/")

text= "the reviewer should do something else"
print("predict :", text)
output = model.predict(text)
for prediction in output:
    print(prediction)