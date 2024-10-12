import openai
import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import random

# Download the required NLTK data
nltk.download('punkt')

# Set the OpenAI API key
openai.api_key = "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"

def generate_response(prompt, model):
    try:
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content
    except openai.OpenAIError as e:
        print(f"Error generating response: {e}")
        return None

def tokenize_data(data):
    try:
        tokenized_data = word_tokenize(data)
        return tokenized_data
    except Exception as e:
        print(f"Error tokenizing data: {e}")
        return None

def create_model(vocab_size):
    model = Sequential()
    model.add(LSTM(64, input_shape=(None, vocab_size)))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, X, y):
    try:
        model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)
    except Exception as e:
        print(f"Error training model: {e}")

def generate_text(model, seed_text, max_length):
    try:
        seed_text = word_tokenize(seed_text)
        seed_text = pad_sequences([seed_text], maxlen=max_length)
        prediction = model.predict(seed_text)
        predicted_word = np.argmax(prediction[0])
        return predicted_word
    except Exception as e:
        print(f"Error generating text: {e}")
        return None

def main():
    # Generate a response from the OpenAI API
    prompt = "Hello!"
    response = generate_response(prompt, "gpt-3.5-turbo")

    # Tokenize the response
    tokenized_response = tokenize_data(response)

    # Create a vocabulary of unique words
    vocab = set(tokenized_response)
    vocab_size = len(vocab)

    # Convert the tokenized response into numerical representations
    word_to_index = {word: i for i, word in enumerate(vocab)}
    index_to_word = {i: word for i, word in enumerate(vocab)}
    numerical_response = [word_to_index[word] for word in tokenized_response]

    # Pad the numerical response to a fixed length
    max_length = 100
    padded_response = pad_sequences([numerical_response], maxlen=max_length)

    # One-hot encode the padded response
    one_hot_response = np.zeros((len(padded_response), max_length, vocab_size))
    for i, sequence in enumerate(padded_response):
        for j, word in enumerate(sequence):
            one_hot_response[i, j, word] = 1

    # Split the one-hot encoded response into input and output sequences
    X = one_hot_response[:, :-1, :]
    y = one_hot_response[:, 1:, :]

    # Create and train the model
    model = create_model(vocab_size)
    train_model(model, X, y)

    # Generate text using the trained model
    seed_text = "Hello"
    generated_text = []
    for _ in range(100):
        predicted_word = generate_text(model, seed_text, max_length)
        generated_text.append(index_to_word[predicted_word])
        seed_text = " ".join(generated_text)

    print("Generated text:")
    print(" ".join(generated_text))

if __name__ == "__main__":
    main()