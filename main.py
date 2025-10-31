import re
import numpy as np
from functions import random_oversample,calculate_metrics,process_position_data,positional_encoding
import pandas as pd
from numpy import asarray
from numpy import zeros
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Model
from tensorflow import metrics
import tensorflow as tf
from tensorflow.keras.layers import Add, Lambda,Embedding,Dense,Concatenate,Bidirectional,LSTM,Multiply,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from keras.backend import clear_session

def create_and_train_lstm_model(vocab_size, embedding_dim, embedding_matrix, train_data,train_X_promise,
                                train_labels,
                                X1_layers, X1_indices,
                                lstm_units=64, batch_size=32, epochs=20,
                                learning_rate=0.001, 
                                ):

    # src_input
    word_input = Input(shape=(None,), name="word_input")  
    layer_input = Input(shape=(None,), name="layer_input")  
    index_input = Input(shape=(None,), name="index_input")  

    word_embedding = Embedding(input_dim=vocab_size,
                                output_dim=embedding_dim,
                                weights=[embedding_matrix],
                                trainable=False,
                                name="word_embedding")(word_input)


    layer_encoding_matrix = positional_encoding(position=X1_layers.shape[1], d_model=embedding_dim)
    layer_embedding = Lambda(lambda x: tf.gather(
        tf.convert_to_tensor(layer_encoding_matrix, dtype=tf.float32),
        tf.cast(x, tf.int32)), name="layer_embedding")(layer_input)
    
    position_encoding_matrix = positional_encoding(position=X1_indices.shape[1], d_model=embedding_dim)
    position_embedding = Lambda(lambda x: tf.gather(
        tf.convert_to_tensor(position_encoding_matrix, dtype=tf.float32),
        tf.cast(x, tf.int32)), name="position_embedding")(index_input)
    

    merged_embedding = Add(name="add_embeddings")([word_embedding, layer_embedding, position_embedding])
    x = Bidirectional(LSTM(lstm_units), name="bilstm")(merged_embedding)
    sce_gate = Dense(lstm_units*2, activation='sigmoid', name='sce_gate')(x )
    sce_gated_res = Multiply(name='sce_gated_res')([sce_gate, x])


    promise_input = Input(shape=(20, 1), name='promise_input')  
    promise_lstm_out = Bidirectional(LSTM(lstm_units),name='promise_lstm')(promise_input)
    promise_gate = Dense(lstm_units*2, activation='sigmoid', name='promise_gate')(promise_lstm_out)
    promise_gated_res = Multiply(name='promise_gated_res')([promise_gate, promise_lstm_out])


    merge = Concatenate()([sce_gated_res, promise_gated_res])

    merge_droput = Dropout(0.25)(merge)
    output = Dense(1, activation='sigmoid', name='main_output')(merge_droput)


    model = Model(inputs=[word_input, layer_input, index_input,promise_input], outputs=output, name="LSTM_with_MLP")

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=BinaryCrossentropy(),
                  metrics=[metrics.BinaryAccuracy(name='accuracy'),
                           metrics.Precision(name='precision'),
                           metrics.Recall(name='recall')])


    history = model.fit([train_data, X1_layers, X1_indices,train_X_promise], 
                        train_labels,
                        batch_size=batch_size,  
                        epochs=epochs
                        ) 

    return model, history
# your project
project = [
    'ant'
]

max_lengths = {'ant': 500}

file_name = '{}'.format(project)

path = r"./dataset/datasets/ant/ant_1.5.csv"
path1 = r"./dataset/datasets/ant/ant_1.6.csv"


df = random_oversample(path, target_column='bug', random_state=42)
df1 = pd.read_csv(path1)

x_train = df.drop('bug', axis=1)
x_train = x_train.drop('file_name', axis=1)
y_train = df['bug']
x_test = df1.drop('bug', axis=1)
x_test = x_test.drop('file_name', axis=1)
y_test = df1['bug'].apply(lambda x: 1 if x > 0 else x)
X1_train = []
sentences = list(x_train["node"])
for sen in sentences:
    X1_train.append(sen)
X1_train = ["" if pd.isnull(text) else text for text in X1_train]

X1_positions_train = []
sentences = list(x_train["position"])
for sen in sentences:
    X1_positions_train.append(sen)
X1_positions_train = ["" if pd.isnull(text) else text for text in X1_positions_train]


X1_test = []
sentences = list(x_test["node"])
for sen in sentences:
    X1_test.append(sen)
X1_test = ["" if pd.isnull(text) else text for text in X1_test]

X1_positions_test = []
sentences = list(x_test["position"])
for sen in sentences:
    X1_positions_test.append(sen)
X1_positions_test = ["" if pd.isnull(text) else text for text in X1_positions_test]

tokenizer = Tokenizer(num_words=6000)
tokenizer.fit_on_texts(X1_train)
maxlen = max_lengths[project[0]]  

word_counts = tokenizer.word_counts
sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1]) 

low_frequency_words = [word for word, _ in sorted_word_counts]

def truncate_to_maxlen(sequence, positions, maxlen, low_frequency_words):

    sequence_list = sequence.split()
    positions_list = re.findall(r'\(.*?\)', positions)

    if len(sequence_list) != len(positions_list):

        raise ValueError("The lengths of sequence and positions do not match")


    filtered_sequence_positions = [
        (word, pos) for word, pos in zip(sequence_list, positions_list) if word in tokenizer.word_index
    ]

    sequence_list, positions_list = zip(*filtered_sequence_positions) if filtered_sequence_positions else ([], [])


    i = 0
    while len(sequence_list) > maxlen and i < len(sequence_list):
        if sequence_list[i] in low_frequency_words:
            sequence_list = sequence_list[:i] + sequence_list[i + 1:]
            positions_list = positions_list[:i] + positions_list[i + 1:]
        else:
            i += 1

    sequence_list = list(sequence_list)  
    positions_list = list(positions_list)
    while len(sequence_list) < maxlen:
        sequence_list.append("unk")
        positions_list.append("(0,0)")

    filtered_sequence_str = " ".join(sequence_list)
    filtered_positions_str = " ".join(positions_list)

    return filtered_sequence_str, filtered_positions_str


X1_train, X1_positions_train = zip(*[
    truncate_to_maxlen(sentence, positions, maxlen, low_frequency_words)
    for sentence, positions in zip(X1_train, X1_positions_train)
])
X1_layers_train, X1_indices_train=process_position_data(X1_positions_train)
X1_layers_train = np.array(X1_layers_train)
X1_indices_train = np.array(X1_indices_train)
scaler1 = MinMaxScaler()
scaler1.fit(X1_layers_train)
X1_layers_train_scaled = scaler1.transform(X1_layers_train)
scaler2 = MinMaxScaler()
scaler2.fit(X1_indices_train)
X1_indices_train_scaled = scaler2.transform(X1_indices_train)

X1_test, X1_positions_test = zip(*[
    truncate_to_maxlen(sentence, positions, maxlen, low_frequency_words)
    for sentence, positions in zip(X1_test, X1_positions_test)
])
X1_layers_test, X1_indices_test = process_position_data(X1_positions_test)
X1_layers_test = np.array(X1_layers_test)
X1_layers_test_scaled = scaler1.transform(X1_layers_test)
X1_indices_test = np.array(X1_indices_test)
X1_indices_test_scaled = scaler2.transform(X1_indices_test)

X1_train = tokenizer.texts_to_sequences(X1_train)
X1_test = tokenizer.texts_to_sequences(X1_test)
vocab_size = len(tokenizer.word_index) + 1
X1_train = pad_sequences(X1_train, padding='post', maxlen=maxlen)
X1_test = pad_sequences(X1_test, padding='post', maxlen=maxlen)

embeddings_dictionary = dict()
embedding_dim=128
model_file = r'./dataset/embedding/ant/ant_1.5_word2vec_embedding.txt'
glove_file = open(model_file, encoding="utf8")
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()        

embedding_matrix = zeros((vocab_size, embedding_dim))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


selected_columns = [
    "wmc", "dit", "noc", "cbo", 'rfc', "lcom", "ca", "ce", "npm", "lcom3",
    "loc", "dam", "moa", "mfa", "cam", "ic", "cbm", "amc", "max_cc", "avg_cc"
]

X2_train = x_train[selected_columns].copy()
X2_test = x_test[selected_columns].copy()
def clean_data(df):
    df.replace('-', np.nan, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df.fillna(df.mean(), inplace=True)
    return df
X2_train = clean_data(X2_train)
X2_test = clean_data(X2_test)
X2_train = X2_train.values
X2_test = X2_test.values
scaler = MinMaxScaler()
scaler.fit(X2_train)
Xtrain_scaled = scaler.transform(X2_train)
Xtest_scaled = scaler.transform(X2_test)

input_1 = Input(shape=(maxlen,))
input_2 = Input(shape=(20, 1))
train_data=X1_train
layer_input=X1_layers_train
position_input=X1_indices_train
test_data=X1_test
results=[]
print('{} started:'.format(file_name))

clear_session()
model,history=create_and_train_lstm_model(
vocab_size=vocab_size,
embedding_matrix=embedding_matrix,
embedding_dim=embedding_dim,
train_data=train_data,
X1_layers= X1_layers_train,
X1_indices= X1_indices_train,
train_X_promise=Xtrain_scaled,
train_labels=y_train,
lstm_units=32,
batch_size=256,
epochs=30,
learning_rate=0.0005)

# model.summary()
pred_prob = model.predict([test_data,X1_layers_test, X1_indices_test,Xtest_scaled])
pred_y = (pred_prob > 0.5).astype(int)  
results = calculate_metrics(y_test, pred_y,results)
print(results)

