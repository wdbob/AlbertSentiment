import os
from albert_sentiment import model, test_data, data_generator
import numpy as np

def gen_output(test_generator):
    l = len(test_generator)
    print('length of test data: ', l)
    cnt = 0
    pred = np.array([]).reshape(0,2)
    label = []
    for x, y in test_generator:
        print(x.shape, y.shape)
        y_pred = model.predict(x)
        cnt += 1
        print(str(cnt)+ '/'+ str(l))
        pred = np.concatenate((pred, y_pred))
        label += y
        print(pred.shape, len(label))


if __name__ == "__main__":

    batch = 1000
    test_generator = data_generator(test_data, batch)
    gen_output(test_generator)