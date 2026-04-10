import torch
import torch.nn as nn
import json
from sentence_transformers import SentenceTransformer
from deep_learning_model import IntentModel

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

epochs = 100

for epoch in range(epochs):
    outputs = model(X)
    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y).sum().item() / len(y)
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}")

torch.save(model.state_dict(), "models/intent_model.pth")
print("Model trained & saved successfully")