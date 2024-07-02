import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return file.read()

def preprocess_text(text, max_len=100):
    chars = sorted(list(set(text)))
    char_to_int = {c: i for i, c in enumerate(chars)}
    int_to_char = {i: c for i, c in enumerate(chars)}
    
    encoded = [char_to_int[c] for c in text]
    vocab_size = len(chars)
    
    sequences = []
    for i in range(0, len(encoded) - max_len, 3):
        sequences.append(encoded[i:i+max_len])
    
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return np.array(padded_sequences), char_to_int, int_to_char, vocab_size, max_len

def create_autoencoder(vocab_size, max_len, latent_dim=64):
    # Encoder
    inputs = Input(shape=(max_len,))
    embedded = Embedding(input_dim=vocab_size, output_dim=32, input_length=max_len)(inputs)
    encoded = LSTM(latent_dim)(embedded)
    
    # Decoder
    decoded = RepeatVector(max_len)(encoded)
    decoded = LSTM(latent_dim, return_sequences=True)(decoded)
    outputs = Dense(vocab_size, activation='softmax')(decoded)
    
    autoencoder = Model(inputs, outputs)
    autoencoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    # Separate encoder model
    encoder = Model(inputs, encoded)
    
    return autoencoder, encoder

def compress(text, char_to_int, encoder, max_len):
    encoded = [char_to_int[c] for c in text]
    padded = pad_sequences([encoded], maxlen=max_len, padding='post')
    return encoder.predict(padded)

def decompress(compressed, autoencoder, int_to_char, max_len):
    decoded = autoencoder.predict(np.zeros((1, max_len)))
    decoded_chars = []
    for i in range(max_len):
        char_index = np.argmax(decoded[0, i])
        if char_index != 0:  # 0 is usually reserved for padding
            decoded_chars.append(int_to_char[char_index])
    return ''.join(decoded_chars)

if __name__ == "__main__":
    input_path = './input.txt'
    text = load_text(input_path)
    padded_sequences, char_to_int, int_to_char, vocab_size, max_len = preprocess_text(text)

    autoencoder, encoder = create_autoencoder(vocab_size, max_len)
    
    # Train the autoencoder
    autoencoder.fit(padded_sequences, padded_sequences, epochs=500, batch_size=32)

    # Compress and decompress
    compressed = compress(text, char_to_int, encoder, max_len)
    decompressed_text = decompress(compressed, autoencoder, int_to_char, max_len)

    print(f"Original text: {text[:100]}")
    print(f"Compressed representation shape: {compressed.shape}")
    print(f"Decompressed text: {decompressed_text[:100]}")

    # Calculate compression ratio
    original_size = len(text.encode('utf-8'))
    compressed_size = compressed.nbytes
    compression_ratio = original_size / compressed_size
    print(f"Compression ratio: {compression_ratio:.2f}")
