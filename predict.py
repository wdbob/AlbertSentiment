import os
from albert_sentiment import model, test_generator

def gen_output():
    for x, y in test_generator:
        y_pred = model.predict(x)
        print(y_pred)
        print(y)

if __name__ == "__main__":
    gen_output()