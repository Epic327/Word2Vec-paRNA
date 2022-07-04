import pandas as pd
import numpy as np
import argparse
import os
import pickle
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


def split(seqs, trainset_num, maxlen, pos_list, neg_list, save=False):
    n_seqs = len(seqs)  # 3790
    n_train = int(0.8 * trainset_num)  # 2400
    n_val = int(0.2 * trainset_num)  # 600
    n_test = n_seqs - n_val - n_train  # 790

    print('整个数据集的个数：%d ' % n_seqs + '\n')
    print('训练集的个数：%d ' % n_train + '\n')
    print('验证集的个数：%d ' % n_val + '\n')
    print('独立测试集的个数：%d ' % n_test)

    X = pad_sequences(seqs, maxlen=maxlen, padding='post')  # 一共n个 pad成3000*600的矩阵,不够600的在后面用0填充
    y = np.array([1] * len(pos_list) + [0] * len(neg_list))  # 返回一个array[1,1,1,1,0,0,0]

    indices = np.arange(n_seqs)  # 生成array([0,1,2,3,4,5,6,.....3789])
    np.random.shuffle(indices)  # 打乱索引

    X = X[indices]  # 打乱数据
    y = y[indices]  # 打乱标签

    X_train = X[0: n_train]
    X_valid = X[n_train: n_val + n_train]
    X_test = X[n_val + n_train:]

    y_train = y[0: n_train]
    y_valid = y[n_train: n_val + n_train]
    y_test = y[n_val + n_train:]

    if save == True:
        with open('./seq_pkl/' + 'X300_k1_s1_train.pkl', 'wb') as pickle_file:  # pickle将对象以文件的 形式存放在磁盘上
            pickle.dump(X_train, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        with open('./seq_pkl/' + 'y300_k1_s1_train.pkl', 'wb') as pickle_file:
            pickle.dump(y_train, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

        with open('./seq_pkl/' + 'X300_k1_s1_valid.pkl', 'wb') as pickle_file:
            pickle.dump(X_valid, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        with open('./seq_pkl/' + 'y300_k1_s1_valid.pkl', 'wb') as pickle_file:
            pickle.dump(y_valid, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

        with open('./seq_pkl/' + 'X300_k1_s1_test.pkl', 'wb') as pickle_file:
            pickle.dump(X_test, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        with open('./seq_pkl/' + 'y300_k1_s1_test.pkl', 'wb') as pickle_file:
            pickle.dump(y_test, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        return X_train, X_valid, X_test, y_train, y_valid, y_test


# 输入数据推荐使用numpy数组，使用list格式输入会报错
def K_Flod_spilt(K, data):
    i = 0
    kf = KFold(n_splits=K)
    for train_index, test_index in kf.split(data):
        train_data = []
        val_data = []
        i += 1
        for index in train_index:
            train_data.append(data[index])
        for index in test_index:
            val_data.append(data[index])

        train = pd.DataFrame(train_data)
        val = pd.DataFrame(val_data)

        # train.to_csv('F:/biodata/enhancedRNA/' + datas + '/5fold-' + str(i) + '-' + 'train.csv', header=None, index=False)
        # val.to_csv('F:/biodata/enhancedRNA/' + datas + '/5fold-' + str(i) + '-' + 'valid.csv', header=None, index=False)


if __name__ == '__main__':
    pos_list = pickle.load(open('./seq_pkl/' + '1k_1s_eRNAlist.pkl', 'rb'))
    neg_list = pickle.load(open('./seq_pkl/' + '1k_1s_mRNAlist.pkl', 'rb'))
    seqs = pos_list + neg_list
    split(seqs, 3000, 300, pos_list, neg_list, True)
