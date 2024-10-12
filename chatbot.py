import openai
import numpy as np
import tensorflow
import nltk
from keras.models import Sequential
from keras.layers import LSTM, Dense
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
nltk.download('punkt')
openai.api_key = "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"
data = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello!"}])

# Tokenize the data
tokenized_data = [word_tokenize(sentence["content"]) for sentence in data]

train_sequences = [word_tokenize(sentence["content"]) for sentence in data]
max_length = 100
padded_data = pad_sequences([np.array(xi) for xi in train_sequences], maxlen=max_length)

# Convert the data into a numerical format
numerical_data = padded_data
model = Sequential()
model.add(LSTM(units=128, input_shape=(max_length, 1)))
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(numerical_data, epochs=10, batch_size=32, validation_split=0.2)

def generate_response(user_input):
    # Tokenize the user input
    user_input_sequence = [word_tokenize(user_input)]
    user_input_padded = pad_sequences([np.array(xi) for xi in user_input_sequence], maxlen=max_length)

    # Generate the response using the LSTM model
    response = model.predict(user_input_padded)
    response = response.reshape((response.shape[1],))

    # Convert the response into a string
    response_words = [str(i) for i in response]
    response = " ".join(response_words)

    return response

# Test the chatbot
user_input = "Hello!"
response = generate_response(user_input)
print(response)