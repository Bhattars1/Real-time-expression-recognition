# This python script is using FER model only in real time

import cv2
import torch
import numpy as np
from prediction.image import predict_FER

# Function to capture video and make FER predictions in real time
def run_fer_realtime():
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit.")
    
    frame_count = 0
    prediction_interval = 100
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame.")
            break

        frame_count += 1

        # Predict expression on every 200th frame
        if frame_count % prediction_interval == 0:
            # print(f"Predicting expression for frame {frame_count}...")
            fer_prediction = predict_FER(frame)
            print(f"\n \n \n Prediction 1: {fer_prediction['top1_label']} \n {fer_prediction['top1_prob']:.4f}")
            print(f"Prediction 2: {fer_prediction['top2_label']} \n {fer_prediction['top2_prob']:.4f}")
        
        # Display the video feed
        cv2.imshow('Real-time FER', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

# Main entry point of the application
if __name__ == "__main__":
    print("Welcome to the Real-time FER System")
    run_fer_realtime()
