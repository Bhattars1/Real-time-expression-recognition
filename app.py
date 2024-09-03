arranging_order = [5, 3, 0, 6, 4, 2, 1]
itol_FER = {0: 'anger', 1: 'sad', 2: 'surprise', 3: 'disgust', 4: 'happy', 5: 'neutral', 6: 'fear'}

import torch
import cv2
import numpy as np
import speech_recognition as sr
from preprocessing.image import preprocessing_pipeline
from preprocessing.text import generate_bert_embeddings
from prediction.image import predict_FER
from prediction.text import predict_text_expression
from prediction.fusion import decision_fusion_prediction
import threading
import queue
import time
import keyboard

# Helper function to capture an image using the webcam
def capture_image():
    print("Opening Camera")
    cap = cv2.VideoCapture(0)
    print("Adjust yourself in front of the camera.")
    print("Press 'spacebar' to capture the image, or 'q' to quit.")

    captured_image = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break

        cv2.imshow('Webcam', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # Capture the image when spacebar is pressed
            captured_image = frame
            print("Image captured!")
            break
        elif key == ord('q'):  # Quit the loop without capturing
            print("Exiting without capturing an image.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_image

# Helper function to capture video from the webcam
def capture_video():
    cap = cv2.VideoCapture(0)
    print("Press 'spacebar' to start/stop recording video.")
    print("Press 'q' to quit.")
    
    frames = []
    recording = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame.")
            break

        cv2.imshow('Video Capture', frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  
            if recording:
                print("Stopping recording...")
                recording = False
                break
            else:
                print("Starting recording...")
                recording = True
                frames = []
        
        if recording:
            frames.append(frame)
        
        if key == ord('q'): 
            print("Exiting without capturing video.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return frames

# Worker thread to capture audio and put it in a queue
def audio_worker(recording_queue, stop_event):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        while not stop_event.is_set():
            try:
                audio = recognizer.listen(source, timeout=1)
                recording_queue.put(audio)
            except sr.WaitTimeoutError:
                pass

# Helper function to capture speech using the microphone
def capture_speech():
    recording_queue = queue.Queue()
    stop_event = threading.Event()
    audio_thread = threading.Thread(target=audio_worker, args=(recording_queue, stop_event))
    audio_thread.start()

    print("Press 'spacebar' to start/stop recording speech.")
    print("Press 'q' to quit.")
    
    recording = False
    audio_data = []

    while True:
        if keyboard.is_pressed(' '):
            if recording:
                print("Stopping recording...")
                stop_event.set()
                audio_thread.join()
                break
            else:
                print("Starting recording...")
                recording = True
                stop_event.clear()
                audio_data = []
                audio_thread = threading.Thread(target=audio_worker, args=(recording_queue, stop_event))
                audio_thread.start()
                time.sleep(1)  # Small delay to ensure the thread starts properly

        if recording:
            if not recording_queue.empty():
                audio_data.append(recording_queue.get())

        if keyboard.is_pressed('q'):
            if recording:
                print("Stopping recording...")
                stop_event.set()
                audio_thread.join()
            print("Exiting without capturing speech.")
            return None

    try:
        print("Recognizing speech...")
        full_audio = sr.AudioData(b''.join([audio.get_wav_data() for audio in audio_data]), 
                                  sr.Microphone().SAMPLE_RATE, 
                                  sr.Microphone().SAMPLE_WIDTH)
        recognizer = sr.Recognizer()
        text = recognizer.recognize_google(full_audio)
        print(f"Captured text: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")

    return None

# Run the Text Expression Recognition (TER) based on user choice of input
def run_ter_only():
    print("Choose an option:")
    print("1. Text input")
    print("2. Speech input")

    choice = input("Enter 1 or 2: ").strip()

    if choice == '1':
        text = input("Enter your text: ").strip()
        if not text:
            print("No text entered. Exiting.")
            return
        
        # Make text prediction
        print("Predicting expression from text...")
        text_prediction = predict_text_expression(text)
        print(f"Top 1 Prediction: {text_prediction['top1_label']} with probability {text_prediction['top1_prob']:.4f}")
        print(f"Top 2 Prediction: {text_prediction['top2_label']} with probability {text_prediction['top2_prob']:.4f}")

    elif choice == '2':
        text = capture_speech()
        if text is None:
            print("No valid speech captured. Exiting.")
            return
        
        # Make speech prediction
        print("Predicting expression from speech...")
        text_prediction = predict_text_expression(text)
        print(f"Top 1 Prediction: {text_prediction['top1_label']} with probability {text_prediction['top1_prob']:.4f}")
        print(f"Top 2 Prediction: {text_prediction['top2_label']} with probability {text_prediction['top2_prob']:.4f}")

    else:
        print("Invalid choice. Exiting.")

# Run the Facial Expression Recognition (FER) based on user choice
def run_fer_only():
    print("Choose an option:")
    print("1. Capture a single image")
    print("2. Capture a video")

    choice = input("Enter 1 or 2: ").strip()

    if choice == '1':
        image = capture_image()
        if image is None:
            print("No image captured. Exiting.")
            return

        # Make predictions
        print("Predicting expressions...")
        image_prediction = predict_FER(image)
        print(f"Top 1 Prediction: {image_prediction['top1_label']} with probability {image_prediction['top1_prob']:.4f}")
        print(f"Top 2 Prediction: {image_prediction['top2_label']} with probability {image_prediction['top2_prob']:.4f}")

    elif choice == '2':
        frames = capture_video()
        if not frames:
            print("No frames captured. Exiting.")
            return

        # Take every third frame
        selected_frames = frames[::3]
        num_frames = len(selected_frames)

        # Initialize a tensor for accumulating probabilities
        total_probs = torch.zeros(len(selected_frames))

        for i, frame in enumerate(selected_frames):
            print(f"Predicting expressions for frame {i + 1}/{num_frames}...")
            frame_prediction = predict_FER(frame)
            total_probs += torch.tensor(frame_prediction['top1_prob']).squeeze()  # Accumulate probabilities using PyTorch tensors

        # Calculate the average probabilities across all frames
        avg_probs = total_probs / num_frames
        avg_probs = avg_probs[arranging_order]
        final_label = torch.argmax(avg_probs)

        # Display the final results
        print("Final prediction based on average probabilities:")
        print(f"Predicted Expression: {itol_FER[final_label.item()]}")
        print(f"Average probabilities: {avg_probs}")

    else:
        print("Invalid choice. Exiting.")

# Run FER, TER, and Fusion prediction with options for text or speech input
def run_fusion():
    print("Choose TER input method:")
    print("1. Text")
    print("2. Speech")

    choice = input("Enter 1 or 2: ").strip()

    if choice == '1':
        text = input("Enter your text: ").strip()
        if not text:
            print("No text entered. Exiting.")
            return

    elif choice == '2':
        text = capture_speech()
        if text is None:
            print("No valid speech captured. Exiting.")
            return
    else:
        print("Invalid choice. Exiting.")
        return

    # Capture video frames
    frames = capture_video()
    if not frames:
        print("No frames captured. Exiting.")
        return

    # Process video frames (take every third frame)
    selected_frames = frames[::3]
    num_frames = len(selected_frames)

    # Initialize a tensor for accumulating probabilities
    total_probs = torch.zeros(len(selected_frames))

    for i, frame in enumerate(selected_frames):
        print(f"Predicting expressions for frame {i + 1}/{num_frames}...")
        frame_prediction = predict_FER(frame)
        total_probs += torch.tensor(frame_prediction['top1_prob']).squeeze()  # Accumulate probabilities using PyTorch tensors

    # Calculate the average probabilities across all frames
    avg_probs = total_probs / num_frames
    avg_probs = avg_probs[arranging_order]
    final_label = torch.argmax(avg_probs)

    # Display the final results
    print("Final prediction based on average probabilities:")
    print(f"Predicted Expression: {itol_FER[final_label.item()]}")
    print(f"Average probabilities: {avg_probs}")

    # Now, run fusion prediction
    print("Running fusion prediction...")
    fusion_result = decision_fusion_prediction(text, frames[0])  # Using the first frame as an example
    print(f"Fused Prediction: {fusion_result['fused_label']} with probability {fusion_result['fused_prob']:.4f}")
    print(f"Second Fused Prediction: {fusion_result['fused_label_2nd']} with probability {fusion_result['fused_prob_2nd']:.4f}")

    print("TER Predictions:")
    print(f"Top 1 Prediction: {fusion_result['text_prediction']['top1_label']} with probability {fusion_result['text_prediction']['top1_prob']:.4f}")
    print(f"Top 2 Prediction: {fusion_result['text_prediction']['top2_label']} with probability {fusion_result['text_prediction']['top2_prob']:.4f}")

    print("FER Predictions:")
    print(f"Top 1 Prediction: {fusion_result['image_prediction']['top1_label']} with probability {fusion_result['image_prediction']['top1_prob']:.4f}")
    print(f"Top 2 Prediction: {fusion_result['image_prediction']['top2_label']} with probability {fusion_result['image_prediction']['top2_prob']:.4f}")

# Main entry point of the application
if __name__ == "__main__":
    print("Welcome to the Emotion Recognition System")
    print("Please select a mode:")
    print("1. Text Expression Recognition (TER)")
    print("2. Facial Expression Recognition (FER)")
    print("3. Fusion (TER + FER)")

    mode = input("Enter 1, 2, or 3: ").strip()

    if mode == '1':
        run_ter_only()
    elif mode == '2':
        run_fer_only()
    elif mode == '3':
        run_fusion()
    else:
        print("Invalid choice. Exiting.")
