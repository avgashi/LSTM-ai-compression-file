### Project Explanation: Text Compression with LSTM Autoencoder

This project focuses on compressing and decompressing text using a Long Short-Term Memory (LSTM) autoencoder model implemented in Python with TensorFlow. The core objective is to minimize the text data size while retaining the ability to reconstruct the original text.

#### Key Components:

1. **Text Preprocessing**:
   - **Loading Text**: The text is read from a file, `input.txt`.
   - **Tokenization**: The text is tokenized into sequences of integers using Keras' `Tokenizer`. Each word or character is converted to a unique integer.
   - **Padding**: The sequences are padded to ensure uniform length, which is required for the LSTM model.

2. **LSTM Autoencoder Model**:
   - **Architecture**: The autoencoder consists of an encoder and a decoder:
     - **Encoder**: Encodes the input sequence into a latent-space representation.
     - **Decoder**: Decodes the latent representation back into the original sequence.
   - **Layers**: The model includes LSTM layers and Dense layers wrapped in `TimeDistributed` to process the sequences.
   - **Compilation**: The model is compiled with the `adam` optimizer and `sparse_categorical_crossentropy` loss function.

3. **Training**:
   - The model is trained on the padded sequences to learn the compression and decompression mappings.

4. **Compression and Decompression Functions**:
   - **Compression**: The function compresses text by encoding the sequences into a latent-space representation using the trained encoder.
   - **Decompression**: The function decompresses the latent representation back into text using the trained decoder.

5. **Handling Shape Mismatches**:
   - Adjustments are made to ensure that the input and output shapes match the expected dimensions for LSTM layers, which expect 3D inputs in the form of `(batch_size, timesteps, features)`.

#### Requirements:

The project relies on the following Python packages:
- `numpy`: For numerical operations.
- `tensorflow`: For building and training the neural network.

#### Example Usage:

1. **Load the Text**: The text file `input.txt` is loaded.
2. **Preprocess the Text**: Tokenize and pad the sequences.
3. **Create and Train the Model**: Define the autoencoder and train it on the preprocessed data.
4. **Compress and Decompress Text**: Use the trained model to compress and decompress the text.

#### Running the Project:

1. Ensure you have the correct versions of the required packages by creating a `requirements.txt` file:
    ```plaintext
    numpy==1.21.6
    tensorflow==2.5.0
    ```
   Install the dependencies using:
    ```bash
    pip install -r requirements.txt
    ```

2. Place the text file (`input.txt`) in the same directory as the script.

3. Run the script using:
    ```bash
    python compress.py
    ```

This project demonstrates how to use deep learning models, specifically LSTM autoencoders, for text compression tasks. It covers data preprocessing, model training, and handling common issues related to input shape mismatches.
