import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load and preprocess text data
def load_text(filepath):
    with open(filepath, 'r') as file:
        text = file.read()
    return text

def preprocess_text(text, max_len=100):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    sequences = tokenizer.texts_to_sequences([text])[0]
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    padded_sequences = pad_sequences([sequences], maxlen=max_len, padding='post')
    return padded_sequences, tokenizer, vocab_size, max_len

# Define LSTM autoencoder
def create_autoencoder(vocab_size, max_len, latent_dim=256):
    inputs = Input(shape=(max_len, 1))
    encoded = LSTM(latent_dim)(inputs)
    decoded = RepeatVector(max_len)(encoded)
    decoded = LSTM(latent_dim, return_sequences=True)(decoded)
    decoded = TimeDistributed(Dense(vocab_size, activation='softmax'))(decoded)
    
    encoder = Model(inputs, encoded)
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return autoencoder, encoder

# Compress and decompress functions
def compress(text, tokenizer, encoder, max_len):
    sequences = tokenizer.texts_to_sequences([text])[0]
    padded_sequences = pad_sequences([sequences], maxlen=max_len, padding='post')
    padded_sequences = np.expand_dims(padded_sequences, -1)  # Add feature dimension
    compressed = encoder.predict(padded_sequences)
    return compressed

def decompress(compressed, autoencoder, tokenizer, max_len, latent_dim=256):
    # Repeat the compressed vector across the time dimension
    repeated_vector = np.repeat(compressed, max_len, axis=0)
    repeated_vector = repeated_vector.reshape((1, max_len, latent_dim))
    decoded = autoencoder.predict(repeated_vector)
    decoded_sequences = np.argmax(decoded, axis=-1)
    words = [tokenizer.index_word.get(idx, '') for idx in decoded_sequences[0]]
    return ' '.join(words).strip()

# Example usage
input_path = './input.txt'
text = load_text(input_path)
padded_sequences, tokenizer, vocab_size, max_len = preprocess_text(text)

# Add feature dimension to the input data
padded_sequences = np.expand_dims(padded_sequences, -1)

autoencoder, encoder = create_autoencoder(vocab_size, max_len)
autoencoder.fit(padded_sequences, np.expand_dims(padded_sequences, -1), epochs=10, batch_size=32)

compressed = compress(text, tokenizer, encoder, max_len)
decompressed_text = decompress(compressed, autoencoder, tokenizer, max_len)

print(f"Original text: {text[:100]}")
print(f"Compressed representation: {compressed}")
print(f"Decompressed text: {decompressed_text[:100]}")
