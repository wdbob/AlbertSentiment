import os
from albert_sentiment import model, test_data, data_generator

def gen_output(test_generator):
    for x, y in test_generator:
        y_pred = model.predict(x)
        print(y_pred)
        print(y)

if __name__ == "__main__":

    batch = 10000
    test_generator = data_generator(test_data, batch)
    gen_output(test_generator)