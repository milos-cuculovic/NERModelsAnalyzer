import sys

from dobert import prediction

model_name = sys.argv[1]
text = sys.argv[2]

print(prediction(text, model_name))

