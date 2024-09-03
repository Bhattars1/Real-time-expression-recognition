from preprocessing.image import preprocessing_pipeline
import torch
import torch.nn as nn
device = "cpu"
itol_FER = {0: 'anger', 1: 'sad', 2: 'surprise', 3: 'disgust', 4: 'happy', 5: 'neutral', 6: 'fear'}


# Create an instance of the model TinyVGG
# Define the model
class TinyVGG(nn.Module):
    def __init__(self, conv1_channels=32, conv2_channels=64, fc1_units=128, dropout_rate=0.5, num_classes=7):
        super(TinyVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, conv1_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(conv2_channels * 16 * 16, fc1_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc1_units, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x

# Create an instance of the model
TinyVGG_model = TinyVGG(conv1_channels=64, conv2_channels = 64, fc1_units=256, dropout_rate=0.5 ).to(device)
TinyVGG_model = TinyVGG_model.to(dtype=torch.float32)

# Getting the keys
TinyVGG_model_state_dict = torch.load("models/TinyVGG.pth", map_location=torch.device('cpu'), weights_only=True)
TinyVGG_model.load_state_dict(TinyVGG_model_state_dict)


# Prediction fuction
def predict_FER(image):
    image = preprocessing_pipeline(image)
    with torch.inference_mode():
        pred_logits = TinyVGG_model(image.permute(2, 0, 1).unsqueeze(dim=0).to(torch.float32).to(device))
        pred_probs = torch.softmax(pred_logits, dim=1).squeeze()
        
        top2_probs, top2_indices = torch.topk(pred_probs, 2)

        top1_label = itol_FER[top2_indices[0].item()]
        top2_label = itol_FER[top2_indices[1].item()]

        return {
            'top1_label': top1_label,
            'top1_prob': top2_probs[0].item(),
            'top2_label': top2_label,
            'top2_prob': top2_probs[1].item()
        }

# def predict_FER(image):
#   image = preprocessing_pipeline(image)
#   with torch.inference_mode():
#     pred_logits = TinyVGG_model(image.permute(2,0,1).unsqueeze(dim=0).to(torch.float32).to(device))

#     # Get the prediction probabilities for our samples
#     pred_probs = torch.softmax(pred_logits, dim=1)
#     label = itol_FER[pred_probs.argmax(dim=1).item()]
#   return label, pred_probs.argmax(dim=1).item(), pred_probs