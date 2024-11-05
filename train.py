import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# Load intents and responses data
with open('intents.json', 'r') as file:
    intents = json.load(file)

all_words = []
tags = []
xy = []

# Process each intent pattern and associate it with its tag
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)  # add unique tag for each intent
    for pattern in intent['patterns']:
        # Tokenize each pattern to get a list of words
        tokenized_pattern = tokenize(pattern)
        all_words.extend(tokenized_pattern)  # add words to vocabulary
        xy.append((tokenized_pattern, tag))  # pair words with the associated tag

# Stem and filter out unnecessary symbols
ignore_words = ['?', '.', '!']
all_words = [stem(word) for word in all_words if word not in ignore_words]
all_words = sorted(set(all_words))  # Remove duplicates and sort
tags = sorted(set(tags))  # Sort tags

# Prepare training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # Convert sentence to bag of words vector
    bow_vector = bag_of_words(pattern_sentence, all_words)
    X_train.append(bow_vector)
    # Label as the index of the tag
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Set hyperparameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)

# Dataset class for loading training data
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Initialize DataLoader for batching
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model, loss function, and optimizer
model = NeuralNet(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'Final loss: {loss.item():.4f}')

# Save model and training metadata
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Training complete. Model saved to {FILE}')

