import pickle
import matplotlib.pyplot as plt
from sklearn import metrics
import tensorflow as tf
from tensorflow import expand_dims


def loss_plot(history, figure_prefix, plot_name):
    fig = plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(loss, label='loss')
    plt.plot(val_loss, label='val_loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(figure_prefix + plot_name)


def acc_plot(history, figure_prefix, plot_name):
    fig = plt.figure()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(acc, label='accuracy')
    plt.plot(val_acc, label='val_accuracy')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(figure_prefix + plot_name)


def roc_curve(y_test, X_test):
    Font = {'size': 18, 'family': 'Times New Roman'}

    cnn1d = tf.keras.models.load_model('./model/cnn-k1-s1-m600-1.h5', custom_objects={'expand_dims': expand_dims})
    cnn2d = tf.keras.models.load_model('./model/2dcnn-k1-s1-m600-1.h5', custom_objects={'expand_dims': expand_dims})
    textcnn1d = tf.keras.models.load_model('./model/textcnn-k1-s1-m600-2.h5',
                                           custom_objects={'expand_dims': expand_dims})
    textcnn2d = tf.keras.models.load_model('./model/text2dcnn-k1-s1-m600-2.h5',
                                           custom_objects={'expand_dims': expand_dims})
    lstm = tf.keras.models.load_model('./model/lstm-k1-s1-m600-2.h5', custom_objects={'expand_dims': expand_dims})
    bilstm = tf.keras.models.load_model('./model/bilstm-k1-s1-m600-4.h5', custom_objects={'expand_dims': expand_dims})
    cnn_bilstm = tf.keras.models.load_model('./model/ab-k1-s1-m600-2.h5', custom_objects={'expand_dims': expand_dims})

    lw = 2
    y_pred1 = cnn1d.predict(X_test)
    y_pred2 = cnn2d.predict(X_test)
    y_pred3 = textcnn1d.predict(X_test)
    y_pred4 = textcnn2d.predict(X_test)
    y_pred5 = lstm.predict(X_test)
    y_pred6 = bilstm.predict(X_test)
    y_pred7 = cnn_bilstm.predict(X_test)

    fpr1, tpr1, thres1 = metrics.roc_curve(y_test, y_pred1)
    fpr2, tpr2, thres2 = metrics.roc_curve(y_test, y_pred2)
    fpr3, tpr3, thres3 = metrics.roc_curve(y_test, y_pred3)
    fpr4, tpr4, thres4 = metrics.roc_curve(y_test, y_pred4)
    fpr5, tpr5, thres5 = metrics.roc_curve(y_test, y_pred5)
    fpr6, tpr6, thres6 = metrics.roc_curve(y_test, y_pred6)
    fpr7, tpr7, thres7 = metrics.roc_curve(y_test, y_pred7)

    roc_auc1 = 0.0
    roc_auc2 = 0.0
    roc_auc3 = 0.0
    roc_auc4 = 0.0
    roc_auc5 = 0.0
    roc_auc6 = 0.0
    roc_auc7 = 0.0
    # roc_auc5 = metrics.auc(fpr5, tpr5)
    # roc_auc6 = metrics.auc(fpr6, tpr6)
    # roc_auc7 = metrics.auc(fpr7, tpr7)

    plt.figure(figsize=(10, 10))

    plt.plot(fpr1, tpr1, 'b', label='1DCNN(AUC = %0.4f)' % roc_auc1, color='Red')
    plt.plot(fpr2, tpr2, 'b', label='2DCNN(AUC = %0.4f)' % roc_auc2, color='darkorange')
    plt.plot(fpr3, tpr3, 'b', label='1DTextCNN(AUC = %0.4f)' % roc_auc3, color='green')
    plt.plot(fpr4, tpr4, 'b', label='2DTextCNN(AUC = %0.4f)' % roc_auc4, color='RoyalBlue')
    plt.plot(fpr5, tpr5, 'b', label='LSTM(AUC = %0.4f)' % roc_auc5, color='yellow')
    plt.plot(fpr6, tpr6, 'b', label='BiLSTM(AUC = %0.4f)' % roc_auc6, color='purple')
    plt.plot(fpr7, tpr7, 'b', label='CNN-BiLSTM(AUC = %0.4f)' % roc_auc7, color='black')

    plt.legend(loc='lower right', prop=Font)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.ylabel('True Positive Rate', Font)
    plt.xlabel('False Positive Rate', Font)
    #     plt.tick_params(labelsize=15)
    plt.title('paRNAs ROC Curve', Font)
    plt.tick_params(labelsize=15)
    #     plt.show()
    plt.savefig("./figure/roc-curve2.png")


if __name__ == '__main__':
    X_test = pickle.load(open('./exper_pkl/' + 'test-x.pkl', 'rb'))
    y_test = pickle.load(open('./exper_pkl/' + 'test-y.pkl', 'rb'))
    roc_curve(y_test, X_test)
