import pickle
import multiprocessing

import numpy as np
from gensim.models.word2vec import LineSentence
import random
import gensim,logging
from keras_preprocessing.sequence import pad_sequences


def load_data(filename):
    """加载数据
    单条格式：(文本, 标签id)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text, label = l.strip().split(',')
            D.append((text, int(label)))
    return D

def shuffle_text(file_input, file_output):
    f = open(file_input)          #打开输入数据
    oo = open(file_output, 'w')   #打开输出数据
    entire_file = f.read()         #读入数据
    file_list = entire_file.split('\n')   #按\n分割返回一个列表
    num_lines = len(file_list)    #列表的长度，即序列的个数
    random_nums = random.sample(range(num_lines), num_lines) #随机生成（0-n）区间所有的数，返回一个列表
    for i in random_nums:  #遍历列表
        oo.write(file_list[i] + "\n")    #将随机数组的每一个数字作为索引，写出索引对应的序列，换行

    oo.close()
    f.close()


def seq2ngram(seqs, k, s, wv):   #如果num《200000 返回的是[[],[],[],[]......，[]]索引
    f = open(seqs)
    lines = f.readlines()  #readlines() 方法用于读取所有行(直到结束符 EOF)并返回列表，该列表可以由 Python 的 for... in ... 结构进行处理。
                                                            # 如果碰到结束符 EOF 则返回空字符串
    f.close()
    list22 = []  #返回每一条序列k-mer对应的索引，构成的列表，列表数为序列的总数
    print('need to n-gram %d lines' % len(lines))   #lines一共多少行
    # f = open(dest, 'w')

    for num, line in enumerate(lines):
        if num < 200000:
            line = line[:-1] # remove '\n' and lower ACGT   去除最后一个换行 把ACGT变成小写
            l = len(line)  # length of line  一条序列的长度
            list2 = []     #所有序列的k-mer构成的列表
            for i in range(0, l, s):  #每一行序列的长度 ，步长s=1 ，k=3
                if i + k >= l + 1:    # k-mer 滑过整条序列
                    break
                list2.append(line[i:i + k]) #取出 k-mer个碱基，加入列表list2
            #     f.write(''.join(line[i:i + k])) #每k-mer个碱基按‘ ’连接起来
            #     f.write(' ')  #write() 方法用于向文件中写入指定字符串。每一个mer加一个空格
            # f.write('\n')
            list22.append(convert_data_to_index(list2, wv))  #把索引和list2中的k-mer对应起来加入一个新的列表list22
    # f.close()
    return list22


def convert_sequences_to_index(list_of_sequences, wv):   #把序列转换成索引
    ll = []   #
    for i in range(len(list_of_sequences)):   #遍历所有的序列
        ll.append(convert_data_to_index(list_of_sequences[i], wv))  #把每一个序列都附上一个索引加入一个新的列表ll
    return ll


def convert_data_to_index(string_data, wv):   #把每一个mer都加上一个索引
    index_data = []
    for word in string_data:    #遍历列表中的每一个字符串   string_data = list2
        if word in wv:   #如果这个字符串在WV ：应该是64种不同的mer  wv = model1.wv
            index_data.append(wv.vocab[word].index)  #就把这个字符串的索引加入一个新的列表index_data[]
    return index_data


def seq2ngram2(seqs, k, s, dest):   #如果num《100000   ，dest:所有序列的k-mer 返回的是pos对应的mer，或者neg对应的mer
    f = open(seqs)
    lines = f.readlines()
    f.close()

    print('need to n-gram %d lines' % len(lines))
    f = open(dest, 'w')
    for num, line in enumerate(lines):
        if num < 100000:
            line = line[:-1]  # remove '\n' and lower ACGT
            l = len(line)  # length of line

            for i in range(0, l, s):
                if i + k >= l + 1:
                    break

                f.write(''.join(line[i:i + k]))
                f.write(' ')
            f.write('\n')
    f.close()


def word2vect(pos_sequences, k, s, vector_dim,  w2v_path):   #阳性序列：pos
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    seq2ngram2(pos_sequences, k, s, './kmer-data/eRNA_seq_k' + str(k) + '_s' + str(s) + '.txt')   #调用seq2ngram2函数把pos序列转换成k-mer
    sentences = LineSentence('./kmer-data/eRNA_seq_k' + str(k) + '_s' + str(s) + '.txt')  #str()转化成字符串

    mode1 = gensim.models.Word2Vec(sentences, iter=20, window=int(18 / s), min_count=50, size=vector_dim,
                                   workers=multiprocessing.cpu_count())     #训练模型，每一个k-mer 对应一个特征向量
    mode1.save(w2v_path + 'w2c_eRNA' + '_k' + str(k) + '_s' + str(s) + '_d' + str(vector_dim))  #保存模型


def pos_neg_list(wv_path,k,s,pos_sequences,neg_sequences):
    w2v_model = gensim.models.Word2Vec.load(wv_path)
    # pos_list :即是包含pos序列个数的子列表，每一个子列表都是由该序列所有的mer的索引构成
    pos_list = seq2ngram(pos_sequences, k, s, w2v_model.wv)
    print('前1个line的 pos index: ', pos_list[:1])
    with open('./seq_pkl/' + str(k) + 'k_' + str(s) + 's_' + 'eRNAlist.pkl', 'wb') as pickle_file:   #pickle将对象以文件的 形式存放在磁盘上
        pickle.dump(pos_list, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    neg_list = seq2ngram(neg_sequences, k, s, w2v_model.wv)
    print('前1个line的 neg index: ', neg_list[:1])
    with open('./seq_pkl/' + str(k) + 'k_' + str(s) + 's_' + 'mRNAlist.pkl', 'wb') as pickle_file:
        pickle.dump(neg_list, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

def csv_to_index_pkl(maxlen,csv_file,wv_path,X_pkl, y_pkl):
    w2v_model = gensim.models.Word2Vec.load('./wv_model/' + wv_path)
    csv = open('./kfold_csv/' + csv_file,'r')
    lines = csv.readlines()
    csv.close()

    index_list = []  # 返回每一条序列k-mer对应的索引，构成的列表，列表数为序列的总数
    label_list = []
    print('need to n-gram %d lines' % len(lines))  # lines一共多少行

    for num, line in enumerate(lines):
        if num < 200000:
            label = line.split(',')[1][:-1]
            label_list.append(label)
            line = line.split(',')[0][:-1]  # remove '\n'    去除最后一个换行
            l = len(line)  # length of line  一条序列的长度
            token_list = []  # 所有序列的k-mer构成的列表
            for i in range(0, l, 2):  # 每一行序列的长度 ，步长s=1 ，k=3
                token_list.append(line[i:i + 1])
            index_list.append(convert_data_to_index(token_list, w2v_model.wv))

    with open('./exper_pkl/' + X_pkl, 'wb') as p1:  # pickle将对象以文件的 形式存放在磁盘上
        X = pad_sequences(index_list, maxlen=maxlen, padding='post')  # 一共n个 pad成2400*600的矩阵,不够600的在后面用0填充
        pickle.dump(X, p1, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./exper_pkl/' + y_pkl, 'wb') as p2:  # pickle将对象以文件的 形式存放在磁盘上
        pickle.dump(np.array(label_list).astype('int32'), p2, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    csv_to_index_pkl(600,'5fold-3-train.csv','w2c_eRNA_k1_s1_d40','train-x-3.pkl','train-y-3.pkl')
    # word2vect('./text-data/paRNA.txt', 5, 1, 40, './wv_model/')
    # pos_neg_list('./wv_model/w2c_paRNA_k5_s1_d40', 5, 1, './text-data/paRNA.txt', './text-data/mRNA.txt')
