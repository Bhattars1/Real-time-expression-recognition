import queue
import threading
import time
import speech_recognition as sr
import torch
from prediction.text import predict_text_expression
import keyboard

# Function to record audio for a specified duration
def record_audio(duration=5):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Recording audio...")
        audio = recognizer.listen(source, timeout=duration)
        print("Recording completed.")
    return audio

# Function to recognize and predict emotion from the recorded audio
def predict_emotion_from_speech(audio):
    try:
        recognizer = sr.Recognizer()
        # Convert audio to text
        text = recognizer.recognize_google(audio)
        print(f"Recognized text: {text}")

        # Predict emotion from text
        prediction = predict_text_expression(text)
        top1_label = prediction['top1_label']
        top1_prob = prediction['top1_prob']
        top2_label = prediction['top2_label']
        top2_prob = prediction['top2_prob']

        print(f"Top 1 prediction: {top1_label} with probability: {top1_prob:.4f}")
        print(f"Top 2 prediction: {top2_label} with probability: {top2_prob:.4f}")

    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")

# Function to handle the main recording and prediction loop
def run_speech_ter():
    print("Press 'spacebar' to start recording audio.")
    print("The system will record for 5 seconds, wait for 5 seconds, and repeat.")

    while True:
        if keyboard.is_pressed(' '):
            print("Starting recording...")
            audio = record_audio(duration=5)  # Record audio for 5 seconds

            print("Predicting...")
            time.sleep(5)  # Simulate time for processing the prediction
            
            # Predict emotion from the recorded speech
            predict_emotion_from_speech(audio)

            print("Recording audio...")
            time.sleep(5)  # Wait for 5 seconds before next recording
            
        if keyboard.is_pressed('q'):
            print("Exiting...")
            break

if __name__ == "__main__":
    run_speech_ter()
