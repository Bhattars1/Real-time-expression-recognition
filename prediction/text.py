from preprocessing.text import generate_bert_embeddings
import torch
import torch.nn as nn
device = "cpu"

# Create an instance of model
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

input_dim = 768
hidden_dim = 128
output_dim = 7
num_layers = 1

language_model = LSTMClassifier(input_dim, hidden_dim, output_dim, num_layers)
language_model.to(device)

# Load the model
language_model_state_dict = torch.load("models/lstm_model.pth", map_location=torch.device('cpu'))
language_model.load_state_dict(language_model_state_dict)

itol_FER = {0: 'anger', 1: 'sad', 2: 'surprise', 3: 'disgust', 4: 'happy', 5: 'neutral', 6: 'fear'}



# Prediction function
def predict_text_expression(text):
  arranging_order = [5, 3, 0, 6, 4, 2, 1]
  with torch.inference_mode():
    embedded_text = generate_bert_embeddings(text).to(device)
    logits = language_model(embedded_text.unsqueeze(0))
    probs = torch.softmax(logits, dim=1).squeeze()
    probs = probs[arranging_order]
    label = probs.argmax(0)
    return itol_FER[label.item()], label.item(), probs