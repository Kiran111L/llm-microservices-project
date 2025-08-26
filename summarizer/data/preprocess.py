preprocess.py
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#  Text cleaning function

def clean_text(text):
    text = text.lower()                          # lowercase
    text = re.sub(r'<.*?>', '', text)            # remove HTML tags
    text = re.sub(r'\([^)]*\)', '', text)        # remove text in brackets
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)   # remove punctuation/symbols
    text = re.sub(r'\s+', ' ', text).strip()     # remove extra spaces
    return text


# 2. Load dataset

def load_dataset(path, text_col, summary_col, num_samples=50000):
    df = pd.read_parquet(path)  # for wikipedia parquet file
    df = df[[text_col, summary_col]].dropna()
    df = df.sample(n=min(num_samples, len(df)), random_state=42)

    # clean
    df['clean_text'] = df[text_col].apply(clean_text)
    df['clean_summary'] = df[summary_col].apply(clean_text)

    # add <start> and <end>
    df['clean_summary'] = df['clean_summary'].apply(lambda x: '<start> ' + x + ' <end>')

    return df


# 3. Tokenization + padding

def tokenize_and_pad(texts, num_words=50000, max_len=300):
    tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return tokenizer, padded

if __name__ == "__main__":
    # Example usage
    dataset_path = "data/wiki_a.parquet"   # change to your parquet file
    df = load_dataset(dataset_path, text_col="text", summary_col="title", num_samples=20000)

    print("Sample article:", df['clean_text'].iloc[0][:300])
    print("Sample summary:", df['clean_summary'].iloc[0])

    # tokenize
    text_tokenizer, text_data = tokenize_and_pad(df['clean_text'].values, max_len=400)
    summary_tokenizer, summary_data = tokenize_and_pad(df['clean_summary'].values, max_len=50)

    # train-test split
    X_train, X_val, y_train, y_val = train_test_split(text_data, summary_data, test_size=0.1, random_state=42)

    print("Text shape:", X_train.shape, "Summary shape:", y_train.shape)
