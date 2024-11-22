
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("lstm_player_model.keras")

# App title and description
st.title("FIFA Player Performance Prediction")
st.write("Predict a player's performance or transfer value based on their attributes.")

# Input fields for player attributes
st.sidebar.header("Player Attributes")
overall = st.sidebar.slider("Overall", min_value=0, max_value=100, value=75)
potential = st.sidebar.slider("Potential", min_value=0, max_value=100, value=80)

# Add more features as needed
pace = st.sidebar.slider("Pace", min_value=0, max_value=100, value=70)
shooting = st.sidebar.slider("Shooting", min_value=0, max_value=100, value=65)
dribbling = st.sidebar.slider("Dribbling", min_value=0, max_value=100, value=75)

# Create a time-series sequence for prediction (dummy example)
sequence_length = 5
lstm_features = [overall, potential, pace, shooting, dribbling]
sequence = np.tile(lstm_features, (sequence_length, 1))  # Repeat the input for a fixed sequence length

# Reshape input for LSTM
sequence = sequence.reshape(1, sequence_length, len(lstm_features))

# Predict using the model
if st.button("Predict"):
    prediction = model.predict(sequence)
    st.write(f"Predicted Value: {prediction[0][0]:.2f}")
