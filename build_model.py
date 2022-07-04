from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Embedding, Conv2D, Conv1D, concatenate, Convolution1D,GRU
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, LSTM, Lambda
from tensorflow.keras.layers import MaxPooling1D, AveragePooling1D, Bidirectional, MaxPooling2D, GlobalMaxPooling2D
from tensorflow import expand_dims


class gru:
    def build_gru(self,embedding_matrix, maxlen):
        model = Sequential()
        model.add(Embedding(input_dim=embedding_matrix.shape[0],

                            output_dim=embedding_matrix.shape[1],

                            weights=[embedding_matrix],

                            input_length=maxlen,

                            trainable=True))

        model.add(GRU(64))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu', name='myfeatures'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        return model


class text2dcnn:
    def build_text2dcnn(self, embedding_matrix, maxlen):
        input = Input(shape=(maxlen,), dtype='float64')
        # 词嵌入
        embedder = Embedding(input_dim=embedding_matrix.shape[0],

                             output_dim=embedding_matrix.shape[1],

                             input_length=maxlen,

                             weights=[embedding_matrix],

                             trainable=False)

        embed = embedder(input)
        embedding = Lambda(lambda x: expand_dims(x, -1))(embed)

        # 词窗大小分别为3，4，5
        cnn1 = Conv2D(64, (3,40), padding='valid', strides=1, activation='relu')(embedding)
        cnn1 = MaxPooling2D((5,1),strides=2)(cnn1)

        cnn2 = Conv2D(64, (4,40), padding='valid', strides=1, activation='relu')(embedding)
        cnn2 = MaxPooling2D((4,1),strides=2)(cnn2)

        cnn3 = Conv2D(64, (5,40), padding='valid', strides=1, activation='relu')(embedding)
        cnn3 = MaxPooling2D((3,1),strides=2)(cnn3)

        cnn = concatenate([cnn1, cnn2, cnn3], axis=1)
        flat = Flatten()(cnn)

        drop1 = Dropout(0.2)(flat)
        # dense1 = Dense(64, activation='relu')(drop1)
        # drop2 = Dropout(0.2)(dense1)
        # dense2 = Dense(32, activation='relu')(drop2)
        output = Dense(1,activation='sigmoid')(drop1)
        model = Model(inputs=input, outputs=output)

        return model



class cnn_2d:
    def build_2dcnn(self, embedding_matrix, maxlen):
        input = Input(shape=(maxlen,), dtype='float64')
        # 词嵌入
        embedder = Embedding(input_dim=embedding_matrix.shape[0],

                             output_dim=embedding_matrix.shape[1],

                             input_length=maxlen,

                             weights=[embedding_matrix],

                             trainable=False)

        embed = embedder(input)
        embedding = Lambda(lambda x: expand_dims(x, -1))(embed)

        cnn1 = Conv2D(64,
                      3,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      )(embedding)  # shape=[batch_size,maxlen-2,256]
        cnn1 = MaxPooling2D(2)(cnn1)  # shape=[batch_size,256]

        cnn2 = Conv2D(32,
                      3,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      )(cnn1)

        cnn2 = MaxPooling2D(2)(cnn2)
        # cnn2 = GlobalMaxPooling2D()(cnn2)  # 可以代替flatten层 + maxPooling
        flat = Flatten()(cnn2)
        drop = Dropout(0.2)(flat)
        dense = Dense(16, activation='relu')(drop)
        output = Dense(1, activation='sigmoid')(dense)
        model = Model(inputs=input, outputs=output)
        return model


class textcnn:
    def build_textcnn(self, embedding_matrix, maxlen):
        # 模型结构：词嵌入 - 卷积池化 * 3 -拼接 - 全连接 - dropout - 全连接
        input = Input(shape=(maxlen,), dtype='float64')
        # 词嵌入
        embedder = Embedding(input_dim=embedding_matrix.shape[0],

                             output_dim=embedding_matrix.shape[1],

                             input_length=maxlen,

                             weights=[embedding_matrix],

                             trainable=False)

        embed = embedder(input)

        # 词窗大小分别为3，4，5
        cnn1 = Conv1D(128, 3, padding='valid', strides=1, activation='relu')(embed)
        cnn1 = MaxPooling1D(5, strides=2)(cnn1)

        cnn2 = Conv1D(128, 4, padding='valid', strides=1, activation='relu')(embed)
        cnn2 = MaxPooling1D(4, strides=2)(cnn2)

        cnn3 = Conv1D(128, 5, padding='valid', strides=1, activation='relu')(embed)
        cnn3 = MaxPooling1D(3, strides=2)(cnn3)

        cnn = concatenate([cnn1, cnn2, cnn3], axis=1)
        flat = Flatten()(cnn)

        drop1 = Dropout(0.2)(flat)
        # dense1 = Dense(64, activation='relu')(drop1)
        # drop2 = Dropout(0.2)(dense1)
        # dense2 = Dense(32, activation='relu')(drop2)
        output = Dense(1, activation='sigmoid')(drop1)

        model = Model(inputs=input, outputs=output)
        return model


class cnn:
    def build_cnn(self, embedding_matrix, maxlen):
        model = Sequential()
        model.add(Embedding(input_dim=embedding_matrix.shape[0],

                            output_dim=embedding_matrix.shape[1],

                            weights=[embedding_matrix],

                            input_length=maxlen,

                            trainable=True))

        model.add(Convolution1D(filters=100,

                                kernel_size=7,

                                activation='relu',

                                padding='valid'))

        model.add(MaxPooling1D(4, 4))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu', name='myfeatures'))
        # model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        return model


class acnn:
    def build_acnn(self, embedding_matrix, maxlen):
        model = Sequential()
        model.add(Embedding(input_dim=embedding_matrix.shape[0],

                            output_dim=embedding_matrix.shape[1],

                            weights=[embedding_matrix],

                            input_length=maxlen,

                            trainable=True))

        model.add(Convolution1D(filters=100,

                                kernel_size=7,

                                activation='relu',

                                padding='valid'))

        model.add(MaxPooling1D(4, 4))
        model.add(Convolution1D(100, 1, activation='relu'))  # 再输入k*1个卷积
        model.add(MaxPooling1D(2, 2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu', name='myfeatures'))
        # model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        return model


class lstm:
    def build_lstm(self, embedding_matrix, maxlen):
        model = Sequential()
        model.add(Embedding(input_dim=embedding_matrix.shape[0],

                            output_dim=embedding_matrix.shape[1],

                            weights=[embedding_matrix],

                            input_length=maxlen,

                            trainable=True))

        model.add(LSTM(128))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu', name='myfeatures'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        return model


class bilstm:
    def build_bilstm(self, embedding_matrix, maxlen):
        model = Sequential()
        model.add(Embedding(input_dim=embedding_matrix.shape[0],

                            output_dim=embedding_matrix.shape[1],

                            weights=[embedding_matrix],

                            input_length=maxlen,

                            trainable=True))

        model.add(Bidirectional(LSTM(100)))
        # model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu', name='myfeatures'))
        # model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        return model


class acnn_bilstm:
    def build_acnn_bisltm(self, embedding_matrix, maxlen):
        model = Sequential()
        model.add(Embedding(input_dim=embedding_matrix.shape[0],

                            output_dim=embedding_matrix.shape[1],

                            weights=[embedding_matrix],

                            input_length=maxlen,

                            trainable=True))

        model.add(Convolution1D(filters=100,

                                kernel_size=7,

                                activation='relu',

                                padding='valid'))

        model.add(MaxPooling1D(4, 4))
        model.add(Convolution1D(100, 1, activation='relu'))  # 再输入k*1个卷积
        model.add(MaxPooling1D(2, 2))
        model.add(Bidirectional(LSTM(100)))
        # model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu', name='myfeatures'))
        # model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        return model

class cnn2d_blstm:
    def build_2dcnn(self, embedding_matrix, maxlen):
        input = Input(shape=(maxlen,), dtype='float64')
        # 词嵌入
        embedder = Embedding(input_dim=embedding_matrix.shape[0],

                             output_dim=embedding_matrix.shape[1],

                             input_length=maxlen,

                             weights=[embedding_matrix],

                             trainable=False)

        embed = embedder(input)
        embedding = Lambda(lambda x: expand_dims(x, -1))(embed)

        cnn1 = Conv2D(32,
                      3,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      )(embedding)  # shape=[batch_size,600-32+1,40,32]
        cnn1 = MaxPooling2D(2)(cnn1)  # shape=[batch_size,256]

        cnn2 = Conv2D(64,
                      3,
                      strides=1,
                      padding='valid',
                      activation='relu',
                      )(cnn1)

        cnn2 = GlobalMaxPooling2D()(cnn2)

        bi = Bidirectional(LSTM(100))(cnn2)

        d1 = Dense(64,activation='relu')(bi)

        drop = Dropout(0.2)(d1)

        d2 = Dense(32,activation='relu')(drop)

        out = Dense(1,activation='sigmoid')(d2)

        model = Model(inputs=input, outputs=out)
        return model
