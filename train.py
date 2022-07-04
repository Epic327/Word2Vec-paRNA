from datetime import datetime
from gensim.models.word2vec import LineSentence
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping,TensorBoard
import argparse
import numpy as np
from tensorflow.keras.optimizers import Adam
import os

from py_helper.plot_curve import *
from py_helper.data_helper import *
from build_model import *
from py_helper.self_metrics import *

# 指定第一块GPU可用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int)
parser.add_argument('--batch', type=int)
parser.add_argument('--maxlen', type=int)
parser.add_argument('--model', type=str, default='acnn-bilstm')
parser.add_argument('--w2v_path', type=str)
parser.add_argument('--out', type=str)
parser.add_argument('--acc_path', type=str)
parser.add_argument('--loss_path', type=str)

args = parser.parse_args()

epochs = args.epochs
batch = args.batch
maxlen = args.maxlen
model_select = args.model
w2v_path = args.w2v_path
out = args.out
acc_path = args.acc_path
loss_path = args.loss_path

#加载 w2v 词嵌入模型
wv_model = gensim.models.Word2Vec.load('./wv_model/' + w2v_path)

#加载数据集
X_train = pickle.load(open('./exper_pkl/' + 'train-x-1.pkl', 'rb'))
X_valid = pickle.load(open('./exper_pkl/' + 'valid-x-1.pkl', 'rb'))

y_train = pickle.load(open('./exper_pkl/' + 'train-y-1.pkl', 'rb'))
y_valid = pickle.load(open('./exper_pkl/' + 'valid-y-1.pkl', 'rb'))

X_test = pickle.load(open('./exper_pkl/' + 'test-x.pkl', 'rb'))
y_test = pickle.load(open('./exper_pkl/' + 'test-y.pkl', 'rb'))


# 构建vocab_szie * vector_dim的0嵌入矩阵 ,vector=40 ,每个token_id被预训练为40维的向量
embedding_matrix = np.zeros((len(wv_model.wv.vocab), 40))

#
for i in range(len(wv_model.wv.vocab)):
    embedding_vector = wv_model.wv[wv_model.wv.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

if model_select == 'cnn':
    cnn = cnn()
    model = cnn.build_cnn(embedding_matrix, maxlen)

elif model_select == 'acnn':
    acnn = acnn()
    model = acnn.build_acnn(embedding_matrix, maxlen)

elif model_select == 'lstm':
    lstm = lstm()
    model = lstm.build_lstm(embedding_matrix, maxlen)

elif model_select == 'gru':
    gru = gru()
    model = gru.build_gru(embedding_matrix, maxlen)

elif model_select == 'bilstm':
    bilstm = bilstm()
    model = bilstm.build_bilstm(embedding_matrix, maxlen)

elif model_select == 'textcnn':
    textcnn = textcnn()
    model = textcnn.build_textcnn(embedding_matrix, maxlen)

elif model_select == '2dcnn':
    _2dcnn = cnn_2d()
    model = _2dcnn.build_2dcnn(embedding_matrix, maxlen)

elif model_select == 'text2dcnn':
    text2dcnn = text2dcnn()
    model = text2dcnn.build_text2dcnn(embedding_matrix, maxlen)
else:
    acnn_bilstm = acnn_bilstm()
    model = acnn_bilstm.build_acnn_bisltm(embedding_matrix, maxlen)



if __name__ == '__main__':
    print(model.summary())

    log_dir = "./logs/fit/" + model_select + '/' + datetime.now().strftime("%Y%m%d-%H_%M_%S")

    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpointer = ModelCheckpoint(
        filepath='./model/' + out,
        verbose=1,
        save_best_only=True)

    earlystopper = EarlyStopping(monitor='val_loss', patience=6, verbose=2)

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(1e-5), metrics=['accuracy'])


    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch, shuffle=True,
                        validation_data=(X_valid, y_valid),
                        callbacks=[checkpointer, earlystopper,tensorboard_callback]
                        )
    acc_plot(history, './figure/', acc_path)
    loss_plot(history, './figure/', loss_path)

    # predict valid and test

    val_acc, val_f1, val_prec, val_recall, val_mcc, val_conf, val_auc = model_predict('./model/' + out, X_valid, y_valid)
    print('val acc: %05f' % val_acc)
    print('val recall: %05f' % val_recall)
    print('val precision: %05f' % val_prec)
    print('val f1: %05f' % val_f1)
    print('val mcc: %05f' % val_mcc)
    print('val auc: %05f' % val_auc)
    print('val c_matrix: ', val_conf)

    print('***************************************')

    test_acc, test_f1, test_prec, test_recall, test_mcc, test_conf, test_auc = model_predict('./model/' + out, X_test, y_test)
    print('test acc: %05f' % test_acc)
    print('test recall: %05f' % test_recall)
    print('test precision: %05f' % test_prec)
    print('test f1: %05f' % test_f1)
    print('test mcc: %05f' % test_mcc)
    print('test auc: %05f' % test_auc)
    print('test c_matrix: ', test_conf)
