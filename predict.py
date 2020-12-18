import os
from albert_sentiment import model, test_data, data_generator
import numpy as np
import pickle

def gen_output(test_generator):
    l = len(test_generator)
    print('length of test data: ', l)
    cnt = 0
    pred = np.array([]).reshape(0,2)
    #label = []
    for x, _ in test_generator:
        y_pred = model.predict(x)
        cnt += 1
        print(str(cnt)+ '/'+ str(l))
        pred = np.concatenate((pred, y_pred))
        #label += y
        print(pred.shape, len(label))
    with open('data/test_pred.pkl', 'wb') as f:
        pickle.dump(pred, f)


if __name__ == "__main__":

    batch = 1000
    test_generator = data_generator(test_data, batch)
    gen_output(test_generator)