from prediction.text import predict_text_expression
from prediction.image import predict_FER

itol_FER = {0: 'anger', 1: 'sad', 2: 'surprise', 3: 'disgust', 4: 'happy', 5: 'neutral', 6: 'fear'}

def decision_fusion_prediction(text, image, text_weight=1, image_weight=1):

  text_label,_, text_probs = predict_text_expression(text)
  image_label,_, image_probs = predict_FER(image)

  # Perform weighted sum of probabilities
  fused_probs = (text_probs * text_weight + image_probs * image_weight) / (text_weight + image_weight)

  fused_label = fused_probs.argmax(dim=-1)

  return itol_FER[fused_label.item()], fused_probs, text_label, text_probs, image_label, image_probs
     
