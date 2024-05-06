import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras import layers
from keras.layers import Input, Embedding, SimpleRNN, Dense

df = pd.read_csv("/content/total_combined.csv")          # read in CSV
df['text'] = df['text'].fillna('')   # get rid of null characters
texts = df['text'].tolist()           # converts them to lists for clf
labels = df['label'].tolist()

max_words = 9000
max_sequence_length = 500

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index

sequences = pad_sequences(sequences, maxlen=max_sequence_length)
labels = np.asarray(labels)

embedding_dim = 64
hidden_units = 32

input_text = Input(shape=(None,))
embedding_layer = Embedding(max_words, embedding_dim)(input_text)
rnn_layer = SimpleRNN(hidden_units)(embedding_layer)
output = Dense(1, activation='sigmoid')(rnn_layer)

model = Model(input_text, output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

start_time = time.time()
model.fit(sequences, labels, epochs=10, batch_size=32, validation_split=0.2)
end_time = time.time()

training_time = end_time - start_time
print("training time: {:.2f} seconds".format(training_time))
