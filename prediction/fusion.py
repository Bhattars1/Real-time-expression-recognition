from prediction.text import predict_text_expression
from prediction.image import predict_FER
import torch
itol_FER = {0: 'anger', 1: 'sad', 2: 'surprise', 3: 'disgust', 4: 'happy', 5: 'neutral', 6: 'fear'}


def decision_fusion_prediction(text, image, text_weight=1, image_weight=1):
    text_prediction = predict_text_expression(text)
    image_prediction = predict_FER(image)

    # Perform weighted sum of probabilities
    fused_probs = (text_prediction['top1_prob'] * text_weight + image_prediction['top1_prob'] * image_weight) / (text_weight + image_weight)
    
    # Get the highest and second-highest probability labels from the fused results
    top2_probs, top2_indices = torch.topk(torch.tensor(fused_probs), 2)

    top1_label = itol_FER[top2_indices[0].item()]
    top2_label = itol_FER[top2_indices[1].item()]

    return {
        'fused_label': top1_label,
        'fused_prob': top2_probs[0].item(),
        'fused_label_2nd': top2_label,
        'fused_prob_2nd': top2_probs[1].item(),
        'text_prediction': text_prediction,
        'image_prediction': image_prediction
    }

# def decision_fusion_prediction(text, image, text_weight=1, image_weight=1):

#   text_label,_, text_probs = predict_text_expression(text)
#   image_label,_, image_probs = predict_FER(image)

#   # Perform weighted sum of probabilities
#   fused_probs = (text_probs * text_weight + image_probs * image_weight) / (text_weight + image_weight)

#   fused_label = fused_probs.argmax(dim=-1)

#   return itol_FER[fused_label.item()], fused_probs, text_label, text_probs, image_label, image_probs
     
