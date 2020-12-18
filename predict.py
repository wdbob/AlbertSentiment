import os
from albert_sentiment import model, test_data, data_generator

def gen_output(test_generator):
    l = len(test_generator)
    print('length of test data: ', l)
    cnt = 0
    for x, y in test_generator:
        y_pred = model.predict(x)
        cnt += len(x)
        print(cnt, '/', l)
        print(type(y), type(y_pred))


if __name__ == "__main__":

    batch = 1000
    test_generator = data_generator(test_data, batch)
    gen_output(test_generator)