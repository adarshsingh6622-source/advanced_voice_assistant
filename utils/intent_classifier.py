from sentence_transformers import SentenceTransformer
import torch
from utils.deep_learning_model import model

embedder = SentenceTransformer('all-MiniLM-L6-v2')
model.load_state_dict(torch.load("models/intent_model.pth"))
model.eval()

def predict_intent(text):
    embedding = embedder.encode([text])
    tensor = torch.tensor(embedding).float()

    output = model(tensor)
    predicted = torch.argmax(output, dim=1).item()

    return predicted