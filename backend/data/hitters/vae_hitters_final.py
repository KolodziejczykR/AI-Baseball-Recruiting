import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        # Decoder
        self.fc2 = nn.Linear(latent_dim, 64)
        self.fc3 = nn.Linear(64, input_dim)
        
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.fc2(z))
        return self.fc3(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss_with_mask(recon_x, x, mu, logvar, mask, beta=1.0):
    """VAE loss function that only considers non-missing values"""
    # Reconstruction loss (MSE) - only on non-missing values
    recon_loss = F.mse_loss(recon_x * mask, x * mask, reduction='sum')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss

def prepare_data_for_vae_final(df, features_to_use, scaler_type='standard'):
    """
    Prepare data for VAE training
    
    Args:
        df: DataFrame with baseball data
        features_to_use: List of feature names to use for VAE
        scaler_type: 'standard' or 'minmax'
    
    Returns:
        scaled_data: Scaled numpy array
        scalers: List of fitted scaler objects (one per feature)
        mask: Boolean mask indicating which values were originally NaN
        original_means: Original means of each feature
        original_stds: Original standard deviations of each feature
    """
    # Select features
    data = df[features_to_use].copy()
    
    # Create mask for NaN values
    mask = data.isnull()
    
    # Store original statistics
    original_means = {}
    original_stds = {}
    scalers = []
    
    # Prepare scaled data
    scaled_data = np.zeros_like(data.values)
    
    for i, feature in enumerate(features_to_use):
        feature_data = data[feature]
        valid_data = feature_data.dropna()
        
        if len(valid_data) > 0:
            # Store original statistics
            original_means[feature] = valid_data.mean()
            original_stds[feature] = valid_data.std()
            
            # Create scaler for this feature
            if scaler_type == 'standard':
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
            
            # Scale only valid data
            scaled_valid = scaler.fit_transform(valid_data.values.reshape(-1, 1)).flatten()
            scalers.append(scaler)
            
            # Fill scaled data: valid values get scaled, NaN gets 0 (mean of scaled data)
            scaled_data[:, i] = 0  # Default to 0 (mean of scaled data)
            valid_indices = feature_data.notna()
            scaled_data[valid_indices, i] = scaled_valid
        else:
            # If no valid data, use identity scaler
            original_means[feature] = 0
            original_stds[feature] = 1
            scalers.append(None)
    
    return scaled_data, scalers, mask, original_means, original_stds

def train_vae_final(vae, train_loader, val_loader, epochs=100, lr=1e-3, device='cpu'):
    """Train the VAE model with proper masking"""
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        vae.train()
        train_loss = 0
        for batch_idx, (data, mask) in enumerate(train_loader):
            data, mask = data.to(device), mask.to(device)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(data)
            
            # Use masked loss
            loss = vae_loss_with_mask(recon_batch, data, mu, logvar, mask)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
        # Validation
        vae.eval()
        val_loss = 0
        with torch.no_grad():
            for data, mask in val_loader:
                data, mask = data.to(device), mask.to(device)
                recon_batch, mu, logvar = vae(data)
                val_loss += vae_loss_with_mask(recon_batch, data, mu, logvar, mask).item()
        
        train_losses.append(train_loss / len(train_loader.dataset))
        val_losses.append(val_loss / len(val_loader.dataset))
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch}: Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
    
    return train_losses, val_losses

def fill_missing_values_with_vae_final(df, features_to_use, latent_dim=10, epochs=100, 
                                     batch_size=32, lr=1e-3, scaler_type='standard', device='cpu'):
    """
    Final VAE imputation that preserves original statistics
    
    Args:
        df: Original DataFrame
        features_to_use: List of feature names to use for VAE
        latent_dim: Dimension of latent space
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        scaler_type: Type of scaler ('standard' or 'minmax')
        device: Device to use ('cpu' or 'cuda')
    
    Returns:
        vae_df: DataFrame with filled values
        vae_model: Trained VAE model
        scalers: List of fitted scalers
    """
    print(f"Preparing data for {len(features_to_use)} features...")
    
    # Prepare data with improved method
    scaled_data, scalers, mask, original_means, original_stds = prepare_data_for_vae_final(
        df, features_to_use, scaler_type
    )
    
    # Convert to tensors
    data_tensor = torch.FloatTensor(scaled_data)
    mask_tensor = torch.BoolTensor(~mask.values)  # Invert mask: True for non-missing values
    
    # Create dataset and dataloaders
    dataset = torch.utils.data.TensorDataset(data_tensor, mask_tensor)
    
    # Handle small datasets
    if len(dataset) < 20:
        # For very small datasets, use all data for training
        train_dataset = dataset
        val_dataset = dataset
        print(f"  Small dataset ({len(dataset)} samples) - using all data for training")
    else:
        # Normal train/val split
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Adjust batch size for small datasets
    adjusted_batch_size = min(batch_size, len(train_dataset))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=adjusted_batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=adjusted_batch_size, shuffle=False)
    
    # Initialize VAE
    vae = VAE(input_dim=len(features_to_use), latent_dim=latent_dim).to(device)
    
    print("Training VAE...")
    train_losses, val_losses = train_vae_final(vae, train_loader, val_loader, epochs, lr, device)
    
    # Fill missing values
    print("Filling missing values...")
    vae.eval()
    with torch.no_grad():
        # Get reconstructions
        recon_data, _, _ = vae(data_tensor.to(device))
        recon_data = recon_data.cpu().numpy()
        
        # Create new DataFrame with filled values
        vae_df = df.copy()
        
        # Only replace NaN values with VAE predictions
        for i, feature in enumerate(features_to_use):
            mask_feature = mask[feature]
            if scalers[i] is not None:
                # Inverse transform the reconstructed values
                recon_feature = recon_data[:, i].reshape(-1, 1)
                recon_original_scale = scalers[i].inverse_transform(recon_feature).flatten()
                
                # Only replace NaN values
                vae_df.loc[mask_feature, feature] = recon_original_scale[mask_feature]
    
    print("VAE imputation complete!")
    return vae_df, vae, scalers

def add_missing_indicators(df):
    """
    Add missing indicators for baseball statistics
    
    Args:
        df: Original DataFrame
    
    Returns:
        df_with_indicators: DataFrame with missing indicators added
    """
    print("Adding missing indicators...")
    
    # Define baseball statistics (excluding hard_hit_p and fielding metrics)
    baseball_stats = [
        'hand_speed_max', 'bat_speed_max', 'rot_acc_max',
        'sixty_time', 'thirty_time', 'ten_yard_time', 'run_speed_max',
        'exit_velo_max', 'exit_velo_avg', 'distance_max', 'sweet_spot_p'
    ]
    
    # Filter to only include features that exist in the dataset
    baseball_stats = [f for f in baseball_stats if f in df.columns]
    
    df_with_indicators = df.copy()
    
    # Add missing indicators for each baseball stat
    for stat in baseball_stats:
        indicator_name = f'missing_{stat}'
        df_with_indicators[indicator_name] = df[stat].isnull().astype(int)
        print(f"  Added {indicator_name}")
    
    # Add total number of missing baseball stats
    missing_counts = df_with_indicators[[f'missing_{stat}' for stat in baseball_stats]].sum(axis=1)
    df_with_indicators['number_of_missing'] = missing_counts
    
    print(f"  Added 'number_of_missing' column")
    print(f"  Total missing indicators added: {len(baseball_stats) + 1}")
    
    return df_with_indicators

def create_vae_hitters_final():
    """
    Create the final VAE-filled hitter dataset
    
    Returns:
        vae_hitters: DataFrame with filled values and missing indicators
    """
    # Load the data
    df = pd.read_csv('clean_hitter_data.csv')
    
    # Add missing indicators
    df_with_indicators = add_missing_indicators(df)
    
    # Define features to use for VAE (excluding hard_hit_p and fielding metrics)
    vae_features = [
        'hand_speed_max', 'bat_speed_max', 'rot_acc_max',
        'sixty_time', 'thirty_time', 'ten_yard_time', 'run_speed_max',
        'exit_velo_max', 'exit_velo_avg', 'distance_max', 'sweet_spot_p'
    ]
    
    # Filter to only include features that exist in the dataset
    vae_features = [f for f in vae_features if f in df_with_indicators.columns]
    
    print(f"\nUsing {len(vae_features)} features for VAE (excluding hard_hit_p and fielding metrics):")
    for feature in vae_features:
        nan_count = df_with_indicators[feature].isnull().sum()
        print(f"  {feature}: {nan_count} NaN values")
    
    # Fill missing values using VAE
    vae_hitters, vae_model, scalers = fill_missing_values_with_vae_final(
        df_with_indicators, vae_features, latent_dim=10, epochs=100, batch_size=32
    )
    
    return vae_hitters

def compare_means(df_original, df_vae, features):
    """
    Compare original and VAE means for all features
    
    Args:
        df_original: Original DataFrame
        df_vae: VAE-filled DataFrame
        features: List of features to compare
    """
    print("\n=== MEAN COMPARISON (Original vs VAE) ===")
    print("=" * 60)
    
    for feature in features:
        if feature in df_original.columns and feature in df_vae.columns:
            # Original statistics (valid data only)
            original_valid = df_original[feature].dropna()
            vae_all = df_vae[feature]
            
            original_mean = original_valid.mean()
            vae_mean = vae_all.mean()
            difference = abs(original_mean - vae_mean)
            
            print(f"\n{feature}:")
            print(f"  Original valid mean: {original_mean:.2f}")
            print(f"  VAE all data mean: {vae_mean:.2f}")
            print(f"  Difference: {difference:.2f}")
            print(f"  Original valid std: {original_valid.std():.2f}")
            print(f"  VAE all data std: {vae_all.std():.2f}")

if __name__ == "__main__":
    print("Creating final VAE-filled hitter dataset...")
    
    # Create the final VAE dataset
    vae_hitters = create_vae_hitters_final()
    
    # Load original data for comparison
    original_df = pd.read_csv('clean_hitter_data.csv')
    
    # Features to compare
    features_to_compare = [
        'hand_speed_max', 'bat_speed_max', 'rot_acc_max',
        'sixty_time', 'thirty_time', 'ten_yard_time', 'run_speed_max',
        'exit_velo_max', 'exit_velo_avg', 'distance_max', 'sweet_spot_p'
    ]
    
    # Compare means
    compare_means(original_df, vae_hitters, features_to_compare)
    
    # Save the result
    vae_hitters.to_csv('vae_hitters_final.csv', index=False)
    print(f"\nFinal VAE dataset saved as 'vae_hitters_final.csv'")
    print(f"Dataset shape: {vae_hitters.shape}")
    
    # Show missing indicators summary
    print(f"\n=== MISSING INDICATORS SUMMARY ===")
    print("=" * 40)
    
    missing_columns = [col for col in vae_hitters.columns if col.startswith('missing_')]
    print(f"Missing indicator columns: {len(missing_columns)}")
    for col in missing_columns:
        missing_count = vae_hitters[col].sum()
        print(f"  {col}: {missing_count} missing values")
    
    print(f"\n'number_of_missing' statistics:")
    print(f"  Mean: {vae_hitters['number_of_missing'].mean():.2f}")
    print(f"  Std: {vae_hitters['number_of_missing'].std():.2f}")
    print(f"  Min: {vae_hitters['number_of_missing'].min()}")
    print(f"  Max: {vae_hitters['number_of_missing'].max()}") 