```python
from nbconvert import PythonExporter
import nbformat


import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources
nltk.download('stopwords')

# Define preprocessing function
def clean_text(text):
    # Convert text to lowercase and remove punctuation
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    
    # Tokenize and stem the words
    words = text.split()
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    
    # Define important spam keywords
    spam_keywords = ['free', 'offer', 'limited', 'claim', 'click', 'buy', 'now', 'only', 'cheap']
    
    # Filter words based on stopwords and keep important spam-related words
    words = [stemmer.stem(word) for word in words if word not in stop_words or word in spam_keywords]
    
    return ' '.join(words)

```

    [nltk_data] Downloading package stopwords to C:\Users\MONICA
    [nltk_data]     PUGAZHENDHI\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    


```python
import torch
import torch.nn as nn

# LSTM Model Definition
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

```


```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import numpy as np

# Load dataset
df = pd.read_csv('mail_data.csv')

# Encode labels (Ham = 0, Spam = 1)
le = LabelEncoder()
df['label'] = le.fit_transform(df['Category'])

# Preprocess messages
df['cleaned_message'] = df['Message'].apply(clean_text)

# Tokenize and build vocabulary
tokenized_messages = [message.split() for message in df['cleaned_message']]
flat_tokens = [word for message in tokenized_messages for word in message]
vocab = {word: idx+2 for idx, (word, _) in enumerate(Counter(flat_tokens).items())}
vocab['<PAD>'] = 0
vocab['<UNK>'] = 1

# Convert text to indices
def text_to_indices(text):
    return [vocab.get(word, 1) for word in text.split()]

# Convert the text into indices for LSTM input
df['text_indices'] = df['cleaned_message'].apply(text_to_indices)

# Padding sequences to the same length
max_length = max(df['text_indices'].apply(len))
df['text_indices'] = df['text_indices'].apply(lambda x: x + [0] * (max_length - len(x)))

# Prepare the dataset for training
X = np.array(df['text_indices'].tolist())
y = np.array(df['label'].tolist())

# Convert to torch tensors
X_tensor = torch.tensor(X, dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.float)

```


```python
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
embedding_dim = 100
hidden_dim = 128
output_dim = 1
batch_size = 64
epochs = 5

# Create model
model = LSTMClassifier(len(vocab), embedding_dim, hidden_dim, output_dim).to(device)

# Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(epochs):
    model.train()
    for i in range(0, len(X_tensor), batch_size):
        x_batch = X_tensor[i:i + batch_size].to(device)
        y_batch = y_tensor[i:i + batch_size].to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(x_batch)
        loss = criterion(preds.squeeze(), y_batch)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

```

    Epoch [1/5], Loss: 0.1327
    Epoch [2/5], Loss: 0.1340
    Epoch [3/5], Loss: 0.1359
    Epoch [4/5], Loss: 0.1378
    Epoch [5/5], Loss: 0.1397
    


```python
def predict_message(message):
    # Preprocess and encode the input message
    cleaned = clean_text(message)
    encoded = [vocab.get(word, 1) for word in cleaned.split()]
    input_tensor = torch.tensor(encoded).unsqueeze(0).to(device).long()

    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.sigmoid(output).item()
        prediction = int(probability > 0.5)  # 0.5 is the threshold
        label = le.inverse_transform([prediction])[0]
        return label, probability

```


```python
from sklearn.model_selection import train_test_split

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Convert test data to torch tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float).to(device)
# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    predictions = []
    true_labels = []
    
    for i in range(0, len(X_test_tensor), batch_size):
        x_batch = X_test_tensor[i:i + batch_size]
        y_batch = y_test_tensor[i:i + batch_size]
        
        # Forward pass
        output = model(x_batch)
        predicted = torch.sigmoid(output).squeeze() > 0.5  # Convert to 0 or 1 (Spam or Ham)
        
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(y_batch.cpu().numpy())

# Calculate accuracy
accuracy = np.mean(np.array(predictions) == np.array(true_labels))
print(f"Accuracy: {accuracy * 100:.2f}%")

```

    Accuracy: 86.64%
    


```python
# Interactive mode to predict user input
while True:
    user_input = input("Enter a message to classify (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    result, prob = predict_message(user_input)
    print(f"📬 Prediction: {result.upper()} (Probability: {prob:.2f})\n")

```

    Enter a message to classify (or 'exit' to quit):  Congratulation we have won a prize
    

    📬 Prediction: SPAM (Probability: 0.53)
    
    

    Enter a message to classify (or 'exit' to quit):  Are you coming tonight
    

    📬 Prediction: HAM (Probability: 0.46)
    
    

    Enter a message to classify (or 'exit' to quit):  exit
    


```python

```
