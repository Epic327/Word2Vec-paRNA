import tensorflow as tf
from py_helper.self_metrics import *
import pickle
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--x',type=str)
parser.add_argument('--y',type=str)
parser.add_argument('--model',type=str)
args = parser.parse_args()

X_path = args.x
y_path = args.y
model_path = args.model

def model_predict(model_path,X_test,y_test):

    model = tf.keras.models.load_model(model_path)
    y_prob = model.predict(X_test)
    print('y_prob: ',y_prob.shape)
    y_pred = (model.predict(X_test) > 0.5).astype('int32')   #二分类，最后一层为sigmoid，(model.predict(x) > 0.5).astype("int32")
                                              #多分类，即最后一层为softmax，用这个np.argmax(model.predict(x), axis=-1)
    print('y_pred: ',y_pred.shape)
    accuracy, f1_score, precision, recall, MCC, confusion, auc = myMetrics(y_test,y_pred,y_prob)

    return accuracy, f1_score, precision, recall, MCC, confusion, auc

if __name__ == '__main__':
    X_test = pickle.load(open('./seq_pkl/' + X_path, 'rb'))
    print('x_test: ', X_test[:1])
    y_test = pickle.load(open('./seq_pkl/' + y_path, 'rb'))
    print('y_test: ', y_test[:3])

    accuracy, f1_score, precision, recall, MCC, confusion, auc = model_predict('./model/'+model_path,X_test,y_test)
    print('test acc: %05f' % accuracy + '\n')
    print('test recall: %05f' % recall + '\n')
    print('test precision: %05f' % precision + '\n')
    print('test f1: %05f' % f1_score + '\n')
    print('test mcc: %05f' % MCC + '\n')
    print('test auc: %05f' % auc + '\n')
    print('test c_matrix: ', confusion)


