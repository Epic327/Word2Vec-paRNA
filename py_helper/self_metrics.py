from sklearn import metrics
import tensorflow as tf
'''
y_test : 真实值
y_pred : 预测的标签        
y_prob : 预测的概率
'''

def model_predict(model_path, X_test, y_test):
    model = tf.keras.models.load_model(model_path)
    y_prob = model.predict(X_test)
    # print('y_prob: ', y_prob.shape)
    # y_pred = model.predict_classes(X_test)
    y_pred = (model.predict(X_test) > 0.5).astype('int32')
    # print('y_pred: ', y_pred.shape)
    accuracy, f1_score, precision, recall, MCC, confusion, auc = myMetrics(y_test, y_pred, y_prob)
    return accuracy, f1_score, precision, recall, MCC, confusion, auc

def myMetrics(y_test, y_pred, y_prob):
    accuracy = metrics.accuracy_score(y_test, y_pred)  # 计算准确度

    f1_score = metrics.f1_score(y_test, y_pred)  # 计算F1得分

    precision = metrics.precision_score(y_test, y_pred)  # 计算精确度

    recall = metrics.recall_score(y_test, y_pred)  # 计算召回率

    confusion = metrics.confusion_matrix(y_test, y_pred)  # 计算混淆矩阵

    MCC = metrics.matthews_corrcoef(y_test, y_pred)  # 计算马修斯

    auc = metrics.roc_auc_score(y_test, y_prob)  # 计算auc值

    return accuracy, f1_score, precision, recall, MCC, confusion, auc
