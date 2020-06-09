# -*- coding: utf-8 -*-
import pickle
import numpy as np
import pandas as pd
from NER.utils import Path, CONSTANTS, load_data
from NER.utils import data_processing
from keras.utils import np_utils, plot_model
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Bidirectional, LSTM, Dense, Embedding, TimeDistributed


# 创建模型
def create_Bi_LSTM(vocab_size, label_size, input_shape, output_dim, n_units,train_code):
    if train_code!="":
        print("你输入的代码为：")
        print(train_code)
        exec(train_code)
    else:
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size + 1, output_dim=output_dim, input_length=input_shape, mask_zero=True))
        model.add(Bidirectional(LSTM(units=n_units, activation='selu',return_sequences=True)))
        model.add(TimeDistributed(Dense(label_size + 1, activation='sigmoid')))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def model_train(train_code):
    print("开始训练")
    input_shape = 60

    input_data = load_data()
    data_processing()
    with open(CONSTANTS[1], 'rb') as f:
        word_dictionary = pickle.load(f)
    with open(CONSTANTS[2], 'rb') as f:
        inverse_word_dictionary = pickle.load(f)
    with open(CONSTANTS[3], 'rb') as f:
        label_dictionary = pickle.load(f)
    with open(CONSTANTS[4], 'rb') as f:
        output_dictionary = pickle.load(f)
    vocab_size = len(word_dictionary.keys())
    label_size = len(label_dictionary.keys())

    # 处理输入数据
    aggregate_function = lambda input: [(word, pos, label) for word, pos, label in
                                        zip(input['word'].values.tolist(),
                                            input['pos'].values.tolist(),
                                            input['tag'].values.tolist())]

    grouped_input_data = input_data.groupby('sent_no').apply(aggregate_function)
    sentences = [sentence for sentence in grouped_input_data]

    x = [[word_dictionary[word[0]] for word in sent] for sent in sentences]
    x = pad_sequences(maxlen=input_shape, sequences=x, padding='post', value=0)
    y = [[label_dictionary[word[2]] for word in sent] for sent in sentences]
    y = pad_sequences(maxlen=input_shape, sequences=y, padding='post', value=0)
    y = [np_utils.to_categorical(label, num_classes=label_size + 1) for label in y]

    train_end = int(len(x) * 0.9)
    train_x, train_y = x[0:train_end], np.array(y[0:train_end])
    test_x, test_y = x[train_end:], np.array(y[train_end:])

    #输入参数
    n_units = 100
    batch_size = 32
    epochs = 10
    output_dim = 20

    # 模型训练
    lstm_model = create_Bi_LSTM(vocab_size, label_size, input_shape, output_dim, n_units,train_code)
    lstm_model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)

    # 模型保存
    model_save_path = CONSTANTS[0]
    lstm_model.save(model_save_path)


    # 测试
    N = test_x.shape[0]
    avg_accuracy = 0
    for start, end in zip(range(0, N, 1), range(1, N+1, 1)):
        sentence = [inverse_word_dictionary[i] for i in test_x[start] if i != 0]
        y_predict = lstm_model.predict(test_x[start:end])
        input_sequences, output_sequences = [], []
        for i in range(0, len(y_predict[0])):
            output_sequences.append(np.argmax(y_predict[0][i]))
            input_sequences.append(np.argmax(test_y[start][i]))

        eval = lstm_model.evaluate(test_x[start:end], test_y[start:end])
        print('Test Accuracy: loss = %0.6f accuracy = %0.2f%%' % (eval[0], eval[1] * 100))
        avg_accuracy += eval[1]
        output_sequences = ' '.join([output_dictionary[key] for key in output_sequences if key != 0]).split()
        input_sequences = ' '.join([output_dictionary[key] for key in input_sequences if key != 0]).split()
        output_input_comparison = pd.DataFrame([sentence, output_sequences, input_sequences]).T
        print(output_input_comparison.dropna())
        print('\n\n')

    avg_accuracy /= N
    print("模型准确率：%.2f%%." % (avg_accuracy * 100))
    acc = float("{0:.2f}".format(avg_accuracy * 100))
    return acc

# print("开始训练")
# model_train()



