import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# =========================================================
# 1. Models (Generator + Critic + Gradient Penalty)
# =========================================================

class Generator(nn.Module):
    def __init__(self, noise_dim, num_classes, feature_dim):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(noise_dim + num_classes, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, feature_dim)
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        x = torch.cat((noise, label_input), dim=1)
        return self.model(x)


class Critic(nn.Module):
    def __init__(self, num_classes, feature_dim):
        super(Critic, self).__init__()

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(feature_dim + num_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, features, labels):
        label_input = self.label_emb(labels)
        x = torch.cat((features, label_input), dim=1)
        validity = self.model(x)
        return validity


def compute_gradient_penalty(critic, real_samples, fake_samples, labels, device):
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    alpha = alpha.expand_as(real_samples)

    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
    interpolates.requires_grad_(True)

    d_interpolates = critic(interpolates, labels)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# =========================================================
# 2. Training Function
# =========================================================

def train_wgan():

    # ----------------------------
    # Configuration
    # ----------------------------
    dataset_path = r'C:/Users/shaik/PycharmProjects/PythonProject/Ganproject-lt/CreditCardFraudSystem/dataset/creditcard.csv'
    epochs = 50
    batch_size = 64
    lr = 1e-4
    b1, b2 = 0.0, 0.9
    lambda_gp = 10
    n_critic = 5
    noise_dim = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ----------------------------
    # Load Dataset
    # ----------------------------
    print("Loading dataset...")
    df = pd.read_csv(dataset_path)

    # Use only Fraud class (Class == 1)
    fraud_df = df[df['Class'] == 1].copy()

    features = fraud_df.drop('Class', axis=1).values
    labels = fraud_df['Class'].values  # all 1

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Save scaler
    joblib.dump(scaler, "scaler.pkl")
    print("Scaler saved as scaler.pkl")

    tensor_x = torch.tensor(features_scaled, dtype=torch.float32)
    tensor_y = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    feature_dim = features.shape[1]

    # ----------------------------
    # Initialize Models
    # ----------------------------
    generator = Generator(noise_dim, 2, feature_dim).to(device)
    critic = Critic(2, feature_dim).to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_C = optim.Adam(critic.parameters(), lr=lr, betas=(b1, b2))

    print("Starting WGAN-GP Training...")

    # =====================================================
    # Training Loop
    # =====================================================

    for epoch in range(epochs):

        for i, (real_samples, labels) in enumerate(dataloader):

            real_samples = real_samples.to(device)
            labels = labels.to(device)
            b_size = real_samples.size(0)

            # ==========================
            # Train Critic
            # ==========================
            optimizer_C.zero_grad()

            z = torch.randn(b_size, noise_dim).to(device)
            fake_samples = generator(z, labels)

            real_validity = critic(real_samples, labels)
            fake_validity = critic(fake_samples.detach(), labels)

            gradient_penalty = compute_gradient_penalty(
                critic, real_samples, fake_samples.detach(), labels, device
            )

            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_C.step()

            # ==========================
            # Train Generator
            # ==========================
            if i % n_critic == 0:
                optimizer_G.zero_grad()

                z = torch.randn(b_size, noise_dim).to(device)
                gen_samples = generator(z, labels)

                g_loss = -torch.mean(critic(gen_samples, labels))

                g_loss.backward()
                optimizer_G.step()

        print(f"[Epoch {epoch+1}/{epochs}]  D_loss: {d_loss.item():.4f}  G_loss: {g_loss.item():.4f}")

    # ----------------------------
    # Save Generator
    # ----------------------------
    torch.save(generator.state_dict(), "fraud_generator.pth")
    print("Generator saved as fraud_generator.pth")


# =========================================================
# 3. Run
# =========================================================

if __name__ == "__main__":
    train_wgan()