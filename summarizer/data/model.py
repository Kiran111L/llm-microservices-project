# model.py
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
from tensorflow.keras.models import Model

# ---------------------------
# Seq2Seq Model (Encoder-Decoder)
# ---------------------------

def build_seq2seq_model(
    vocab_size_text, vocab_size_summary,
    embedding_dim=300, lstm_units=256,
    max_text_len=400, max_summary_len=50
):
    # ----- Encoder -----
    encoder_inputs = Input(shape=(max_text_len,))
    encoder_embedding = Embedding(vocab_size_text, embedding_dim, mask_zero=True)(encoder_inputs)
    encoder_lstm, state_h, state_c = LSTM(lstm_units, return_state=True)(encoder_embedding)

    # ----- Decoder -----
    decoder_inputs = Input(shape=(max_summary_len,))
    decoder_embedding = Embedding(vocab_size_summary, embedding_dim, mask_zero=True)(decoder_inputs)
    decoder_lstm, _, _ = LSTM(lstm_units, return_sequences=True, return_state=True)(
        decoder_embedding, initial_state=[state_h, state_c]
    )
    decoder_dense = Dense(vocab_size_summary, activation="softmax")
    decoder_outputs = decoder_dense(decoder_lstm)

    # ----- Model -----
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
