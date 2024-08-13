
import torch
import cv2
from preprocessing.image import preprocessing_pipeline
from preprocessing.text import generate_bert_embeddings
from prediction.image import predict_FER
from prediction.text import predict_text_expression


arranging_order = [5, 3, 0, 6, 4, 2, 1]

import torch
import cv2
from preprocessing.image import preprocessing_pipeline
from preprocessing.text import generate_bert_embeddings
from prediction.image import predict_FER
from prediction.text import predict_text_expression
from prediction.fusion import decision_fusion_prediction

def capture_image():
    """Capture an image using the webcam."""
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

def main():
    # Capture image
    print("Capturing image...")
    image = capture_image()
    if image is None:
        print("No image captured. Exiting.")
        return

    # Ask for text input
    text = input("Enter text: ")
    
    # Make predictions
    print("Predicting expressions...")
    
    # Image prediction
    image_label, _, image_probs = predict_FER(image)
    print(f"Image prediction: {image_label}")
    
    # Text prediction
    text_label, text_label_id, text_probs = predict_text_expression(text)
    print(f"Text prediction: {text_label}")
    

    # Fusion prediction
    fused_label, fused_probs, _, _, _, _ = decision_fusion_prediction(text, image)
    print(f"Fused prediction: {fused_label}")

    # Display probabilities
    print("\nProbabilities:")
    print(f"Image probabilities: {image_probs}")
    print(f"Text probabilities: {text_probs}")
    print(f"Fused probabilities: {fused_probs}")

if __name__ == "__main__":
    main()