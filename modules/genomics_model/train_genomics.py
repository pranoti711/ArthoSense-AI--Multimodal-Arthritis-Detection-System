import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from modules.genomics_model.model import GenomicsModel

# SNPs
snps = ["rs2476601", "rs1801133", "HLA-B27", "PTPN22"]

def generate_synthetic_data(n_samples=2000):
    """
    Generate synthetic SNP data with realistic RA risk labels.
    Risk increases with number of risk alleles.
    """
    X = np.random.randint(0, 2, size=(n_samples, len(snps)))
    # Risk score: sigmoid( sum(risk_alleles) * scale )
    # Assume higher SNPs indicate higher risk
    sum_snps = np.sum(X, axis=1)
    logit = sum_snps * 1.5 - 2.0  # Scale to get good spread
    y = 1 / (1 + np.exp(-logit))  # Sigmoid
    return X, y

def train_model():
    print("🧬 Training Genomics RA Risk Model")
    
    # Generate data
    X, y = generate_synthetic_data(2000)
    print(f"Generated {len(X)} synthetic samples")
    print(f"X shape: {X.shape}, y range: [{y.min():.3f}, {y.max():.3f}]")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    # Model
    model = GenomicsModel(input_dim=len(snps))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Train
    epochs = 200
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = criterion(model(X_test_t), y_test_t)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")
    
    # Save
    model_path = os.path.join(os.path.dirname(__file__), "../../mlops/model_registry/genomics_ra_model.pt")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Test on high risk
    high_risk = torch.tensor([[1,1,1,1]], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred = model(high_risk).item()
    print(f"High risk (all 1s) prediction: {pred:.4f}")
    
    return model

if __name__ == "__main__":
    train_model()

