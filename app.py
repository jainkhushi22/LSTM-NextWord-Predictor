import numpy as np 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import streamlit as st

#load the LSTM Model
model=load_model("next_word_LSTM.h5")

# lode the tokenizer
with open('tokenizer.pickle',"rb") as handle:
    tokenizer=pickle.load(handle)

#function to predict the next word

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding='pre'
    )

    predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word

    return None

# streamlit app
st.title("Next word prediction with LSTM")
input_text=st.text_input("Enter the sequence of words","To be or not to be")
if st.button("predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1


    next_word = predict_next_word(
        model,
        tokenizer,
        input_text,
        max_sequence_len
    )


    st.write("Next Word:", next_word)
