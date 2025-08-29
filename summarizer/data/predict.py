# predict.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from preprocess import clean_text, tokenize_and_pad
import pickle

# ---------------------------
# Load trained model & tokenizers
# ---------------------------
model = load_model("saved_model/seq2seq.h5")

# load saved tokenizers (you should save them after training)
with open("saved_model/text_tokenizer.pkl", "rb") as f:
    text_tokenizer = pickle.load(f)

with open("saved_model/summary_tokenizer.pkl", "rb") as f:
    summary_tokenizer = pickle.load(f)

max_text_len = 400
max_summary_len = 50

# reverse dictionary to convert index -> word
reverse_summary_index = {v: k for k, v in summary_tokenizer.word_index.items()}

# ---------------------------
# Function to generate summary
# ---------------------------
def decode_sequence(input_seq):
    # Encode input as state vectors
    encoder_input = np.array(input_seq).reshape(1, -1)
    
    # Start with <start> token
    target_seq = np.zeros((1, max_summary_len))
    target_seq[0, 0] = summary_tokenizer.word_index['<start>']
    
    summary = []
    for i in range(1, max_summary_len):
        # Predict next word
        output_tokens = model.predict([encoder_input, target_seq], verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, i-1, :])
        sampled_word = reverse_summary_index.get(sampled_token_index, None)
        
        if sampled_word == '<end>' or sampled_word is None:
            break
        summary.append(sampled_word)
        
        target_seq[0, i] = sampled_token_index
    
    return " ".join(summary)

# ---------------------------
# Test with new text
# ---------------------------
def summarize_text(text):
    # clean & preprocess
    clean = clean_text(text)
    seq = text_tokenizer.texts_to_sequences([clean])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_text_len, padding='post')
    return decode_sequence(padded)

if __name__ == "__main__":
    test_article = """Tim Cook is the CEO of Apple since 2011. 
    He succeeded Steve Jobs and previously served as Chief Operating Officer. 
    He has overseen product launches like the iPhone X and Apple Watch."""
    
    print("Input article:", test_article)
    print("Generated summary:", summarize_text(test_article))
