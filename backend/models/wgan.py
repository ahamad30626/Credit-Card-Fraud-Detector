import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=2, feature_dim=29):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        
        # In Conditional GAN, we embed the label and concatenate with noise
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(noise_dim + num_classes, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, feature_dim)
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and noise
        c = self.label_emb(labels)
        x = torch.cat([noise, c], 1)
        return self.model(x)

class Critic(nn.Module):
    def __init__(self, num_classes=2, feature_dim=29):
        super(Critic, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(feature_dim + num_classes, 256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Subtly important: No sigmoid activation at the end for WGAN
            nn.Linear(64, 1)
        )

    def forward(self, features, labels):
        c = self.label_emb(labels)
        x = torch.cat([features, c], 1)
        return self.model(x)

def compute_gradient_penalty(critic, real_samples, fake_samples, labels, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size(0), 1)).to(device)
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    d_interpolates = critic(interpolates, labels)
    
    fake = torch.ones(d_interpolates.size()).to(device)
    
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
