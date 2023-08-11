import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Concatenate, Embedding, Dropout


def load_data_frame(path: str):
    return pd.read_csv(path, sep='\t')


def preprocess_sentences(df, tokenizer):
    sequences_A = tokenizer.texts_to_sequences(df['sentence_A'])
    sequences_B = tokenizer.texts_to_sequences(df['sentence_B'])

    sequences_A = pad_sequences(sequences_A)
    sequences_B = pad_sequences(sequences_B)

    return sequences_A, sequences_B


def create_model(vocab_size, embedding_dim):
    input_A = Input(shape=(None,))
    input_B = Input(shape=(None,))

    embedding_layer = Embedding(vocab_size, embedding_dim)
    shared_lstm = LSTM(64)

    embedded_A = embedding_layer(input_A)
    embedded_B = embedding_layer(input_B)

    encoded_A = shared_lstm(embedded_A)
    encoded_B = shared_lstm(embedded_B)

    merged_vector = Concatenate(axis=-1)([encoded_A, encoded_B])
    merged_vector = Dropout(0.5)(merged_vector)
    predictions = Dense(1)(merged_vector)

    model = Model(inputs=[input_A, input_B], outputs=predictions)
    model.compile(optimizer='adam', loss='mse')
    return model
