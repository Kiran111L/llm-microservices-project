# train.py
import numpy as np
import tensorflow as tf
from preprocess import load_dataset, tokenize_and_pad
from model import build_seq2seq_model
from sklearn.model_selection import train_test_split


# 1. Load and preprocess data

dataset_path = "data/wiki_a.parquet"  # change to your parquet file
df = load_dataset(dataset_path, text_col="text", summary_col="title", num_samples=20000)

# tokenize
text_tokenizer, text_data = tokenize_and_pad(df['clean_text'].values, max_len=400)
summary_tokenizer, summary_data = tokenize_and_pad(df['clean_summary'].values, max_len=50)

# train-test split
X_train, X_val, y_train, y_val = train_test_split(text_data, summary_data, test_size=0.1, random_state=42)

print("Data shapes ->", X_train.shape, y_train.shape)

# 2. Build Model

vocab_size_text = len(text_tokenizer.word_index) + 1
vocab_size_summary = len(summary_tokenizer.word_index) + 1

model = build_seq2seq_model(
    vocab_size_text=vocab_size_text,
    vocab_size_summary=vocab_size_summary,
    embedding_dim=256,
    lstm_units=256,
    max_text_len=400,
    max_summary_len=50
)

model.summary()


# 3. Prepare decoder output

# In training, we shift the summary tokens by one position for teacher forcing
decoder_input_data = y_train[:, :-1]
decoder_output_data = y_train[:, 1:]
decoder_input_val = y_val[:, :-1]
decoder_output_val = y_val[:, 1:]

# expand dims to match sparse_categorical_crossentropy requirement
decoder_output_data = np.expand_dims(decoder_output_data, -1)
decoder_output_val = np.expand_dims(decoder_output_val, -1)


# 4. Train Model

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "saved_model/seq2seq.h5", save_best_only=True, monitor="val_loss", mode="min"
)

history = model.fit(
    [X_train, decoder_input_data],
    decoder_output_data,
    epochs=10,
    batch_size=64,
    validation_data=([X_val, decoder_input_val], decoder_output_val),
    callbacks=[checkpoint]
)

print("✅ Training complete. Best model saved at saved_model/seq2seq.h5")
