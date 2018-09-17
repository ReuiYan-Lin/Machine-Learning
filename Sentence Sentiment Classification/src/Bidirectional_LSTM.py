import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Embedding, Input,InputLayer,BatchNormalization, Dense, Bidirectional,LSTM,Dropout,GRU,Activation
from keras.models import Model,load_model
from keras.callbacks import History ,ModelCheckpoint, EarlyStopping
import pandas as pd
import gensim
import re
from collections import Counter
from keras.utils.generic_utils import get_custom_objects
from keras import regularizers
from keras import backend as K
import pickle


train_label_path = "../data/training_label.csv"
model_path = "../model/Bidirectional_LSTM.h5"
with open(train_label_path,encoding="utf8") as f:
    train = f.readlines()
train_X = [seg.strip().split(" +++$+++ ")[1] for seg in train]
train_y = [seg.strip().split(" +++$+++ ")[0] for seg in train]


stemmer = gensim.parsing.porter.PorterStemmer()
def preprocess(string, use_stem = True):
    string = string.replace("i ' m", "im").replace("you ' re","youre").replace("didn ' t","didnt")    .replace("can ' t","cant").replace("haven ' t", "havent").replace("won ' t", "wont")    .replace("isn ' t","isnt").replace("don ' t", "dont").replace("doesn ' t", "doesnt")    .replace("aren ' t", "arent").replace("weren ' t", "werent").replace("wouldn ' t","wouldnt")    .replace("ain ' t","aint").replace("shouldn ' t","shouldnt").replace("wasn ' t","wasnt")    .replace(" ' s","s").replace("wudn ' t","wouldnt").replace(" .. "," ... ")    .replace("couldn ' t","couldnt")
    for same_char in re.findall(r'((\w)\2{2,})', string):
        string = string.replace(same_char[0], same_char[1])
    for digit in re.findall(r'\d+', string):
        string = string.replace(digit, "1")
    for punct in re.findall(r'([-/\\\\()!"+,&?\'.]{2,})',string):
        if punct[0:2] =="..":
            string = string.replace(punct, "...")
        else:
            string = string.replace(punct, punct[0])
    return string



def token_counter(corpus):
    tokenizer = Tokenizer(num_words=None,filters="\n")
    tokenizer.fit_on_texts(corpus)
    sequences = tokenizer.texts_to_sequences(corpus)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))


train_X_cleaned = [preprocess(sent) for sent in train_X]
train_X_cleaned = [sent for sent in stemmer.stem_documents(train_X_cleaned)]
token_counter(train_X_cleaned)


total_corpus = train_X_cleaned
total_corpus = [ sent.split(" ") for sent in total_corpus]
print(total_corpus)

w2v_model = gensim.models.Word2Vec(total_corpus, size=300, window=5, min_count=0, workers=8)

vocab_size = None
tokenizer = Tokenizer(num_words=vocab_size,filters="\n\t")
tokenizer.fit_on_texts(train_X_cleaned)
sequences = tokenizer.texts_to_sequences(train_X_cleaned)
word_index = tokenizer.word_index
pickle.dump(tokenizer, open('tokenizer.pickle', 'wb'))

print('Found %s unique tokens.' % len(word_index))

oov_count = 0
embedding_matrix = np.zeros((len(word_index), len(word_index)))
for word, i in word_index.items():
    try:
        embedding_vector = w2v_model.wv[word]
        embedding_matrix[i] = embedding_vector
    except:
        oov_count +=1
        print(word)
#         print(word," not in w2v model")
print("oov count: ", oov_count)
max_length = np.max([len(i) for i in sequences])
print("max length:", max_length)
train_X_num = pad_sequences(sequences, maxlen=max_length)
train_y_cat = to_categorical(np.asarray(train_y))
train_y_numeric = np.array(train_y,dtype=int)





embedding_layer = Embedding(len(word_index),output_dim= len(word_index),
                            weights=[embedding_matrix],
                            input_length=max_length,
                            trainable=False)
print("embedd matrix shape: ",embedding_matrix.shape)


batch_size=64
def swish(x):
    return (K.sigmoid(x) * x)
get_custom_objects().update({'swish': Activation(swish)})
for i in range(20):
    if i==0:
        numb_train = 180000
        train_X, valid_X = train_X_num[:numb_train], train_X_num[numb_train:]
        train_y, valid_y = train_y_cat[:numb_train], train_y_cat[numb_train:]
    elif i==1:
        numb_train = -180000
        train_X, valid_X = train_X_num[numb_train:], train_X_num[:numb_train]
        train_y, valid_y = train_y_cat[numb_train:], train_y_cat[:numb_train]
    else:
        train_X, valid_X, train_y, valid_y = train_test_split(train_X_num, train_y_cat, 
                                                              test_size=0.07)

    sequence_input = Input(shape=(max_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    lstm1 = Bidirectional(LSTM(128,activation="tanh",dropout=0.2,return_sequences = True,
                kernel_initializer='he_uniform'))(embedded_sequences)
    lstm2 = Bidirectional(LSTM(64,activation="tanh",dropout=0.2,return_sequences = False,
                kernel_initializer='he_uniform'))(lstm1)
    bn1 = BatchNormalization()(lstm2)
    dense1 = Dense(64, activation="sigmoid")(bn1)
    dropout1 = Dropout(0.5)(dense1)
    dense2 = Dense(32, activation="sigmoid")(dropout1)
    dropout2 = Dropout(0.5)(dense2)
    preds = Dense(2, activation='softmax')(dropout2)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    hist = History()
    check_save  = ModelCheckpoint("../model/modelW2V-{epoch:05d}-{val_acc:.5f}.h5",monitor='val_acc',save_best_only=True,save_weights_only=True)
    early_stop = EarlyStopping(monitor="val_acc", patience=2, verbose=1, mode='max')
    print("model fitting")
    model.summary()
    model.fit(train_X, train_y, validation_data=(valid_X, valid_y),
            epochs=20, batch_size=batch_size,callbacks=[check_save,hist,early_stop])
    model.save(model_path)