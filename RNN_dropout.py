from keras.callbacks import EarlyStopping
from keras.layers import Dropout

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
dropout_rate = 0.2

input_text = Input(shape=(max_sequence_length,))
embedding_layer = Embedding(max_words, embedding_dim)(input_text)
dropout_layer = Dropout(dropout_rate)(embedding_layer)
rnn_layer = SimpleRNN(hidden_units)(dropout_layer)
output = Dense(1, activation='sigmoid')(rnn_layer)

model = Model(input_text, output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

start_time = time.time()
model.fit(sequences, labels, epochs=10, batch_size=32, validation_split=0.2)
end_time = time.time()

training_time = end_time - start_time
print("training time: {:.2f} seconds".format(training_time))
