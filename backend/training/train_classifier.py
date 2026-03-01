import sys
import os

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from models.classifier import FraudClassifier


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset path (CHANGE if needed)
dataset_path = r"C:\Users\shaik\PycharmProjects\PythonProject\Ganproject-lt\CreditCardFraudSystem\dataset\creditcard.csv"

data = pd.read_csv(dataset_path)

X = data.drop("Class", axis=1)
y = data["Class"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
scaler_path = os.path.join(os.path.dirname(__file__), "..", "scaler.pkl")
joblib.dump(scaler, scaler_path)

# Train / Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Convert to tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train.values).view(-1, 1)

X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val.values).view(-1, 1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=256)

# Model
model = FraudClassifier(input_dim=X_train.shape[1]).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 30
best_recall = 0

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        preds = model(xb)
        loss = criterion(preds, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Validation
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            preds = model(xb)

            all_preds.extend(preds.cpu().numpy())
            all_true.extend(yb.numpy())

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    threshold = 0.3
    binary_preds = (all_preds > threshold).astype(int)

    precision = precision_score(all_true, binary_preds)
    recall = recall_score(all_true, binary_preds)
    f1 = f1_score(all_true, binary_preds)
    roc_auc = roc_auc_score(all_true, all_preds)

    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"Loss: {total_loss:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("-" * 40)

    # Save best model
    if recall > best_recall:
        best_recall = recall
        model_path = os.path.join(os.path.dirname(__file__), "..", "fraud_model.pth")
        torch.save(model.state_dict(), model_path)
        print("Model improved and saved!")

print("\nTraining Complete")
print("Best Recall:", best_recall)