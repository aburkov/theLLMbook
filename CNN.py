import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import numpy as np
import gzip
import urllib.request
from tqdm import tqdm
import re

def load_word_vectors():

    filename = "GoogleNews-vectors-negative300.bin.gz"
    url = f"https://github.com/aburkov/theLLMbook/releases/download/v1.0.0/{filename}"

    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=filename) as progress_bar:
        def report_hook(count, block_size, total_size):
            if total_size != -1:
                progress_bar.total = total_size
            progress_bar.update(block_size)

        urllib.request.urlretrieve(url, filename, reporthook=report_hook)

    with gzip.open(filename, 'rb') as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())

        vectors = {}
        binary_len = np.dtype('float32').itemsize * vector_size

        with tqdm(total=vocab_size, desc="Loading word vectors") as pbar:
            for _ in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        word = b''.join(word).decode('utf-8')
                        break
                    if ch != b'\n':
                        word.append(ch)

                vector = np.frombuffer(f.read(binary_len), dtype='float32')
                if re.search(r"^[a-z]+$", word):
                    vectors[word] = vector
                pbar.update(1)

    return vectors

word_vectors = load_word_vectors()
newsgroups = fetch_20newsgroups(remove=("headers", "footers", "quotes"))
X = newsgroups.data
y = newsgroups.target

X_train, X_test, y_train, y_test = train_test_split(X, y,\
                test_size=0.2, random_state=42, shuffle=True)

def embed_text(text, word_vectors, max_length=5000):
    words = text.lower().split()[:max_length]
    embeddings = [word_vectors.get(word, np.zeros(300)) for word in words]
    padding = [np.zeros(300)] * (max_length - len(embeddings))
    return np.array(embeddings + padding)[:max_length]

class NewsGroupDataset(Dataset):
    def __init__(self, texts, labels, word_vectors, max_len):
        self.texts = texts
        self.labels = labels
        self.word_vectors = word_vectors

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        embeddings = embed_text(self.texts[idx], self.word_vectors, max_len)
        return torch.tensor(embeddings, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

class TextCNN(nn.Module):
    def __init__(self, embedding_dim, num_classes, max_len):
        super(TextCNN, self).__init__()
        self.conv1 = nn.Conv1d(embedding_dim, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(512, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * max_len, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Create datasets and dataloaders
max_len = 500
train_dataset = NewsGroupDataset(X_train, y_train, word_vectors, max_len)
test_dataset = NewsGroupDataset(X_test, y_test, word_vectors, max_len)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Initialize the model, loss function, and optimizer
model = TextCNN(300, 20, max_len)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    train_correct = 0
    train_total = 0
    for batch_embeddings, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_embeddings)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        train_total += batch_labels.size(0)
        train_correct += (predicted == batch_labels).sum().item()

    train_accuracy = 100 * train_correct / train_total

    # Evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch_embeddings, batch_labels in test_loader:
            outputs = model(batch_embeddings)
            _, predicted = torch.max(outputs.data, 1)
            test_total += batch_labels.size(0)
            test_correct += (predicted == batch_labels).sum().item()

    test_accuracy = 100 * test_correct / test_total
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

print("Training completed!")
