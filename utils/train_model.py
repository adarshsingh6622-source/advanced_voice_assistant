import torch
import torch.nn as nn
import json
from sentence_transformers import SentenceTransformer
from utils.deep_learning_model import IntentModel

model = IntentModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

embedder = SentenceTransformer('all-MiniLM-L6-v2')

with open("utils/dataset.json") as f:
    data = json.load(f)

texts = [d["text"] for d in data]
labels = [d["label"] for d in data]

embeddings = embedder.encode(texts)
X = torch.tensor(embeddings).float()
y = torch.tensor(labels)

for epoch in range(50):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")

torch.save(model.state_dict(), "models/intent_model.pth")