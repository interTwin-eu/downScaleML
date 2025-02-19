# Standard library imports
import sys
import os
import time
import logging
import subprocess
import pathlib
from datetime import datetime, timedelta
from logging.config import dictConfig

# Data processing and ML libraries
import numpy as np
import pandas as pd
import xarray as xr
import dask
from dask.distributed import LocalCluster, Client
import optuna
from optuna.integration import LightGBMPruningCallback
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

# PyTorch-related imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.autograd import grad
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchmetrics import MeanMetric
from torch.utils.tensorboard import SummaryWriter
from pytorch_msssim import MS_SSIM

from torch.autograd import grad
from torchvision.models import vgg16  # For perceptual loss

# Image processing
from skimage.metrics import structural_similarity

# Visualization
import matplotlib.pyplot as plt

# MLflow
import mlflow
import mlflow.pytorch
from mlflow import MlflowClient

# Downscaleml imports
from downscaleml.core.hpo import objective
from downscaleml.core.dataset import (
    ERA5Dataset, NetCDFDataset, EoDataset, DatasetStacker, 
    SummaryStatsCalculator, doy_encoding
)
from downscaleml.core.clustering import ClusteringWorkflow
from downscaleml.main.config_reanalysis_dev_no_slide import (
    ERA5_PATH, OBS_PATH, DEM_PATH, MODEL_PATH, TARGET_PATH,
    NET, ERA5_PLEVELS, ERA5_PREDICTORS, PREDICTAND,
    CALIB_PERIOD, VALID_PERIOD, DOY, NORM,
    OVERWRITE, DEM, DEM_FEATURES, STRATIFY, WET_DAY_THRESHOLD,
    VALID_SIZE, start_year, end_year, CHUNKS, TEST_PERIOD, RESULT_PERIOD
)
from downscaleml.core.constants import (
    ERA5_P_VARIABLES, ERA5_P_VARIABLES_SHORTCUT, ERA5_P_VARIABLE_NAME,
    ERA5_S_VARIABLES, ERA5_S_VARIABLES_SHORTCUT, ERA5_S_VARIABLE_NAME,
    ERA5_VARIABLES, ERA5_VARIABLE_NAMES, ERA5_PRESSURE_LEVELS,
    PREDICTANDS
)
from downscaleml.core.utils import (
    NAMING_Model, normalize, search_files, LogConfig
)
from downscaleml.core.logging import log_conf
from einops import rearrange

from torchvision.models import vgg16  # For perceptual loss

from torch.optim.lr_scheduler import CosineAnnealingLR


LOGGER = logging.getLogger(__name__)

# module level logger
LOGGER = logging.getLogger(__name__)
client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
mlflow.set_tracking_uri('http://127.0.0.1:5000')
base_dir = "/home/sdhinakaran/downScaleML/downscaleml/main/mlartifacts"

from einops import rearrange

# Transformer Block for Discriminator
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

# HAT Generator (Unchanged)
class Generator(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, embed_dim=64, num_transformers=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        
        self.transformer_blocks = nn.ModuleList([TransformerBlock(embed_dim) for _ in range(num_transformers)])
        
        self.conv_out = nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)  # Initial feature extraction
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')  # Flatten for transformer
        for block in self.transformer_blocks:
            x = block(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)  # Restore spatial shape
        x = self.conv_out(x)
        return x

# Transformer Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=1, embed_dim=64, num_transformers=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(embed_dim) for _ in range(num_transformers)])
        self.fc_out = nn.Linear(embed_dim, 1)
    
    def forward(self, x):
        x = self.conv1(x)  # Initial feature extraction
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')  # Flatten for transformer
        for block in self.transformer_blocks:
            x = block(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc_out(x)
        return x

class MeteorologicalDataset(Dataset):
    def __init__(self, X, y):
        """
        Parameters:
        X (Tensor): Input tensor of shape (lat, lon, time, features).
        y (Tensor): Target tensor of shape (lat, lon, time, features).
        """
        self.X = X  # Shape: (lat, lon, time, features_in)
        self.y = y  # Shape: (lat, lon, time, features_out=1)
        self.lat, self.lon, self.time, self.features = X.shape

    def __len__(self):
        return self.time  # One sample per time step

    def __getitem__(self, idx):
        # Input: All features at the current time step
        input_sample = self.X[:, :, idx, :]  # Shape: (lat, lon, features_in)
        input_sample = input_sample.permute(2, 0, 1)  # Shape: (features_in, lat, lon)

        # Target: Single feature at the current time step
        target_sample = self.y[:, :, idx, :]  # Shape: (lat, lon, 1)
        target_sample = target_sample.permute(2, 0, 1).squeeze(0)  # Shape: (lat, lon)

        return input_sample, target_sample

def prepare_data(X_tensor, y_tensor, batch_size):
    dataset = MeteorologicalDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True, prefetch_factor=2)
    return dataloader

class SSIMLoss(nn.Module):
    def __init__(self, data_range=1.0):
        """
        SSIM Loss using pytorch_msssim.
        Args:
            data_range (float): The range of the input data (e.g., 1.0 for normalized data, 255 for images).
        """
        super(SSIMLoss, self).__init__()
        self.ssim = MS_SSIM(data_range=data_range, size_average=True, channel=1)  # Assuming single-channel output

    def forward(self, x, y):
        """
        Compute SSIM loss between two tensors.
        Args:
            x (torch.Tensor): Predicted tensor of shape [B, C, H, W].
            y (torch.Tensor): Target tensor of shape [B, C, H, W].
        Returns:
            torch.Tensor: SSIM loss (1 - SSIM).
        """
        # Ensure inputs are 4D
        if x.dim() == 3:  # If input is [B, H, W], add channel dimension
            x = x.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
            y = y.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
            
        return 1 - self.ssim(x, y)  # SSIM is in [0,1], so loss = 1 - SSIM



class CombinedLoss(nn.Module):
    def __init__(self, lambda1=1.0, lambda2=0.1, lambda3=0.001, lambda4=0.5, lambda5=0.2, lambda_gp=10.0):
        super(CombinedLoss, self).__init__()
        self.lambda1 = lambda1  # Content loss weight (L1)
        self.lambda2 = lambda2  # Perceptual loss weight
        self.lambda3 = lambda3  # Adversarial loss weight
        self.lambda4 = lambda4  # MS-SSIM loss weight
        self.lambda5 = lambda5  # Frequency loss weight
        self.lambda_gp = lambda_gp  # Gradient penalty weight

        # Define loss functions
        self.l1_loss = nn.L1Loss()  # Content loss (L1)
        self.l2_loss = nn.MSELoss()  # Content loss (L2)
        self.ms_ssim_loss = MS_SSIM()  # MS-SSIM loss
        self.vgg = self._load_vgg()  # VGG for perceptual loss
        self.frequency_loss = nn.L1Loss()  # Frequency loss

    def _load_vgg(self):
        """Load and configure the VGG16 model for perceptual loss."""
        vgg = vgg16(pretrained=True).features[:16]  # Use first 16 layers
        for param in vgg.parameters():
            param.requires_grad = False  # Freeze VGG parameters
        return vgg

    def compute_perceptual_loss(self, fake_data, real_data):
        """Compute perceptual loss using VGG16."""
        fake_features = self.vgg(fake_data)
        real_features = self.vgg(real_data)
        return self.l1_loss(fake_features, real_features)

    def compute_frequency_loss(self, fake_data, real_data):
        """Compute frequency loss using FFT."""
        fft_gen = torch.fft.fft2(fake_data)
        fft_gt = torch.fft.fft2(real_data)
        return self.frequency_loss(torch.abs(fft_gen), torch.abs(fft_gt))

    def compute_gradient_penalty(self, critic, real_samples, fake_samples):
        """Calculates the gradient penalty for WGAN-GP."""
        batch_size = real_samples.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1, device=real_samples.device)
        interpolates = (epsilon * real_samples + (1 - epsilon) * fake_samples).requires_grad_(True)
        critic_interpolates = critic(interpolates)
        
        # Check for NaNs in critic outputs
        if torch.isnan(critic_interpolates).any():
            raise ValueError("Critic output contains NaNs!")

        grad_outputs = torch.ones_like(critic_interpolates, device=real_samples.device).detach()
        gradients = grad(
            outputs=critic_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        grad_norm = gradients.norm(2, dim=1)
        grad_norm = torch.clamp(grad_norm, 0.0, 10.0)
        gradient_penalty = ((grad_norm - 1) ** 2).mean()
        return gradient_penalty

    def forward(self, real_data, fake_data, critic, fake_preds, real_preds):
        """Computes the total loss for Generator and Discriminator."""
        
        if torch.isnan(fake_preds).any() or torch.isnan(real_preds).any():
            raise ValueError("Discriminator predictions contain NaNs!")
        
        # Adversarial Loss (WGAN-GP)
        g_loss_adv = -torch.mean(fake_preds)
        d_loss_real = -torch.mean(real_preds)
        d_loss_fake = torch.mean(fake_preds)

        # Content Loss (L1)
        g_loss_l1 = self.l1_loss(fake_data, real_data)

        # Perceptual Loss (VGG-based)
        perceptual_loss = self.compute_perceptual_loss(fake_data, real_data)

        # MS-SSIM Loss
        ms_ssim_loss = 1 - self.ms_ssim_loss(fake_data, real_data)  # MS-SSIM returns a similarity score (1 - score for loss)

        # Frequency Loss
        frequency_loss = self.compute_frequency_loss(fake_data, real_data)

        # Gradient Penalty for discriminator
        gradient_penalty = self.compute_gradient_penalty(critic, real_data.unsqueeze(1), fake_data.unsqueeze(1))

        # Generator total loss
        g_loss = (
            self.lambda1 * g_loss_l1 +  # Content loss (L1)
            self.lambda2 * perceptual_loss +  # Perceptual loss
            self.lambda3 * g_loss_adv +  # Adversarial loss
            self.lambda4 * ms_ssim_loss +  # MS-SSIM loss
            self.lambda5 * frequency_loss  # Frequency loss
        )
        
        # Discriminator total loss
        d_loss = d_loss_real + d_loss_fake + self.lambda_gp * gradient_penalty  
        
        return g_loss, d_loss


def train_esrgan(
    dataloader_train, 
    dataloader_test,
    generator,
    discriminator,
    optimizer_G,
    optimizer_D,
    scheduler_G,
    scheduler_D,
    device='cuda',
    num_epochs=100,
    patience=25,
    min_delta=0.001,
    lambda1=1.0,  # Content loss (L1) weight
    lambda2=0.1,  # Perceptual loss weight
    lambda3=0.001,  # Adversarial loss weight
    lambda4=0.5,  # MS-SSIM loss weight
    lambda5=0.2,  # Frequency loss weight
    lambda_gp=10.0,  # Gradient penalty weight
    n_critic=5,  # Number of critic iterations per generator iteration
    save_samples_every=5,  # Save samples every N epochs
    save_all_every=5  # Save all models every N epochs
):
    
    # Initialize MLflow experiment
    mlflow.set_experiment("HAT_expr_tasmean_128")
    
    # Create main save directory
    save_dir = f"/mnt/CEPH_PROJECTS/InterTwin/Climate_Downscaling/two_stage/models/tasmean/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Loss functions
    criterion = CombinedLoss(lambda1, lambda2, lambda3, lambda4, lambda5, lambda_gp).to(device)
    ssim_loss = SSIMLoss().to(device)
    
    best_ssim = -float('inf')
    epochs_no_improve = 0

    scaler_G = GradScaler("cuda")
    scaler_D = GradScaler("cuda")
    
    with mlflow.start_run():
        mlflow.log_params({
            "num_epochs": num_epochs,
            "lambda_adv": lambda_adv,
            "lambda_l1": lambda_l1,
            "lambda_l2": lambda_l2,
            "lambda_gp": lambda_gp,
            "n_critic": n_critic,
            "optimizer_G": type(optimizer_G).__name__,
            "optimizer_D": type(optimizer_D).__name__,
        })
        
        # Log model architectures
        mlflow.log_text(str(generator), "generator_architecture.txt")
        mlflow.log_text(str(discriminator), "discriminator_architecture.txt")

        for epoch in range(num_epochs):
            # Create a dedicated directory for current epoch artifacts
            epoch_save_dir = os.path.join(save_dir, f"epoch_{epoch+1}")
            os.makedirs(epoch_save_dir, exist_ok=True)

            LogConfig.init_log(f'Epoch : {epoch}')
            # Training phase
            generator.train()
            discriminator.train()
            
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            total_samples = 0

            total_batches = len(dataloader_train)
            
            for batch_idx, (input_data, target_data) in enumerate(dataloader_train):
                LogConfig.init_log(f"Processing batch {batch_idx + 1}/{total_batches}")
                # Move data to device
                input_data = input_data.float().to(device)  # [B, C, H, W]
                target_data = target_data.float().to(device)  # [B, H, W]
                batch_size = input_data.size(0)
                
                # Train Discriminator (WGAN-GP)
                for _ in range(n_critic):
                    optimizer_D.zero_grad()
                    
                    with autocast("cuda", enabled=False):
                        fake_data = generator(input_data).detach()
                        # Check for NaNs in generator output
                        if torch.isnan(fake_data).any():
                            raise ValueError("Generator output contains NaNs in train discriminator!")
                            
                        real_preds = discriminator(target_data.unsqueeze(1))
                        fake_preds = discriminator(fake_data.unsqueeze(1))

                        _, d_loss = criterion(target_data, fake_data, discriminator, fake_preds, real_preds)

                    scaler_D.scale(d_loss).backward()
                    scaler_D.unscale_(optimizer_D)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                    scaler_D.step(optimizer_D)
                    scaler_D.update()
                
                # -----------------
                #  Train Generator
                # -----------------
                # Train Generator
                optimizer_G.zero_grad()
                
                with autocast("cuda", enabled=False):
                    fake_data = generator(input_data)
                    
                    if torch.isnan(fake_data).any():
                        raise ValueError("Generator output contains NaNs in train discriminator!")
                        
                    fake_preds = discriminator(fake_data.unsqueeze(1))
                    real_preds = discriminator(target_data.unsqueeze(1))  # Ensure consistency
                    
                    g_loss, _ = criterion(target_data, fake_data, discriminator, fake_preds, real_preds)

                scaler_G.scale(g_loss).backward()
                scaler_G.unscale_(optimizer_G)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                scaler_G.step(optimizer_G)
                scaler_G.update()

                epoch_g_loss += g_loss.item() * batch_size
                epoch_d_loss += d_loss.item() * batch_size
                total_samples += batch_size

                
                # Update metrics
                epoch_g_loss += g_loss.item() * batch_size
                epoch_d_loss += d_loss.item() * batch_size
                total_samples += batch_size
            
            # Calculate epoch metrics
            avg_g_loss = epoch_g_loss / total_samples
            avg_d_loss = epoch_d_loss / total_samples
            
            # Validation phase
            generator.eval()
            val_metrics = {
                'mae': 0.0,
                'mse': 0.0,
                'psnr': 0.0,
                'ssim': 0.0,
            }
            
            with torch.no_grad():
                for input_val, target_val in dataloader_test:
                    input_val = input_val.float().to(device)  # [B, C, H, W]
                    target_val = target_val.float().to(device)  # [B, H, W]
                    
                    preds = generator(input_val)  # [B, H, W]
                    
                    # Basic metrics
                    val_metrics['mae'] += F.l1_loss(preds, target_val).item() * input_val.size(0)
                    mse = F.mse_loss(preds, target_val).item()
                    val_metrics['mse'] += mse * input_val.size(0)
                    val_metrics['psnr'] += 10 * np.log10(1 / (mse + 1e-10)) * input_val.size(0)
                    val_metrics['ssim'] += (1 - ssim_loss(preds, target_val).item()) * input_val.size(0)
            
            # Average validation metrics
            for k in val_metrics:
                val_metrics[k] /= len(dataloader_test.dataset)
            
            if (epoch + 1) % save_samples_every == 0:
                sample_input = input_val[0].cpu().numpy()
                sample_pred = preds[0].cpu().numpy()
                sample_target = target_val[0].cpu().numpy()
                
                fig = plot_meteo_samples(
                    input_seq=sample_input,
                    prediction=sample_pred,
                    target=sample_target,
                    variable_type='tasmean',
                    max_val=305
                )
                sample_path = os.path.join(epoch_save_dir, f"samples_epoch_{epoch+1}.png")
                fig.savefig(sample_path)
                plt.close(fig)  # Prevent memory leaks
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                "train_g_loss": avg_g_loss,
                "train_d_loss": avg_d_loss,
                "val_mae": val_metrics['mae'],
                "val_mse": val_metrics['mse'],
                "val_psnr": val_metrics['psnr'],
                "val_ssim": val_metrics['ssim'],
                "lr_generator": optimizer_G.param_groups[0]['lr'],
                "lr_discriminator": optimizer_D.param_groups[0]['lr']
            }, step=epoch)
            
            # Save model checkpoints to epoch directory
            gen_path = os.path.join(epoch_save_dir, f"generator_epoch_{epoch+1}.pth")
            disc_path = os.path.join(epoch_save_dir, f"discriminator_epoch_{epoch+1}.pth")
            torch.save(generator.state_dict(), gen_path)
            torch.save(discriminator.state_dict(), disc_path)

            if (epoch + 1) % save_all_every == 0:
                checkpoint = {
                    'epoch': epoch,
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'optimizer_G_state_dict': optimizer_G.state_dict(),
                    'optimizer_D_state_dict': optimizer_D.state_dict(),
                    'scheduler_G_state_dict': scheduler_G.state_dict(),
                    'scheduler_D_state_dict': scheduler_D.state_dict(),
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state_all(),
                }
                checkpointer = os.path.join(epoch_save_dir, f"checkpoint_{epoch+1}.pth")
                torch.save(checkpoint, checkpointer)
            
            # Log only the current epoch's artifacts
            mlflow.log_artifacts(epoch_save_dir, artifact_path=f"epoch_{epoch+1}")
            
            current_ssim = val_metrics['ssim']
            if current_ssim > best_ssim + min_delta:
                best_ssim = current_ssim
                epochs_no_improve = 0
                # Save best model to main directory
                torch.save(generator.state_dict(), os.path.join(save_dir, "best_generator.pth"))
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Update schedulers (still using MAE unless instructed otherwise)
            scheduler_G.step()
            scheduler_D.step()
    
        # Final cleanup and model logging (unchanged)
        mlflow.log_artifact(os.path.join(save_dir, "best_generator.pth"))
        mlflow.pytorch.log_model(generator, "final_generator")
        mlflow.pytorch.log_model(discriminator, "final_discriminator")
        
    return generator, discriminator


def stacker(xarray_dataset):
    # stack along the lat and lon dimensions
    stacked = xarray_dataset.stack()
    dask_arr = stacked.to_array().data
    xarray_dataset = dask_arr.T
    LogConfig.init_log('Shape is in (spatial, time, variables):{}'.format(xarray_dataset.shape))
    return xarray_dataset

def plot_meteo_samples(input_seq, prediction, target, variable_type='tasmean', max_val=305):
    """
    Universal meteorological visualization function
    Args:
        input_seq: Input sequence [C, H, W] (C=4: [temp, dem, sin_doy, cos_doy])
        prediction: Model prediction [H, W]
        target: Ground truth [H, W]
        variable_type: 'pr' (precipitation), 'tasmean' (temperature), or 'ssrd' (solar radiation)
        max_val: Maximum value for color scaling
    Returns:
        matplotlib.figure.Figure
    """
    # Configure based on variable type
    config = {
        'pr': {
            'cmap': 'viridis',
            'unit': 'mm',
            'diff_range': (-max_val/2, max_val/2),
            'title': 'Precipitation'
        },
        'tasmean': {
            'cmap': 'coolwarm',
            'unit': '°C',
            'diff_range': (-max_val/2, max_val/2),
            'title': 'Temperature'
        },
        'ssrd': {
            'cmap': 'plasma',
            'unit': 'W/m²',
            'diff_range': (-max_val/2, max_val/2),
            'title': 'Solar Radiation'
        }
    }[variable_type]

    fig = plt.figure(figsize=(18, 10), facecolor='white')
    gs = fig.add_gridspec(2, 5)
    axs = [
        fig.add_subplot(gs[0, :2]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[0, 4]),
        fig.add_subplot(gs[1, :])
    ]

    # Plot input variable (first channel is the target variable)
    input_var = input_seq[0]  # [H, W] (temperature or precipitation)
    
    vmin = 250 if variable_type == 'tasmean' else -max_val
    vmax = 305 if variable_type == 'tasmean' else max_val
    
    im0 = axs[0].imshow(input_var, cmap=config['cmap'], vmin=vmin, vmax=vmax)
    axs[0].set_title(f'Input {config["title"]}')
    fig.colorbar(im0, ax=axs[0], shrink=0.8, label=config['unit'])

    # Target plot
    im1 = axs[1].imshow(target, cmap=config['cmap'], vmin=vmin, vmax=vmax)
    axs[1].set_title(f'True {config["title"]}')
    fig.colorbar(im1, ax=axs[1], shrink=0.8, label=config['unit'])

    # Prediction plot
    im2 = axs[2].imshow(prediction, cmap=config['cmap'], vmin=vmin, vmax=vmax)
    axs[2].set_title(f'Predicted {config["title"]}')
    fig.colorbar(im2, ax=axs[2], shrink=0.8, label=config['unit'])

    # Difference plot
    diff = prediction - target
    im3 = axs[3].imshow(diff, cmap='coolwarm',
                        vmin=config['diff_range'][0],
                        vmax=config['diff_range'][1])
    axs[3].set_title(f'Prediction Error ({config["unit"]})')
    fig.colorbar(im3, ax=axs[3], shrink=0.8)

    # Metrics panel
    mae = np.mean(np.abs(diff))
    mse = np.mean(diff**2)
    ssim = structural_similarity(target, prediction, data_range=vmax - vmin, win_size=7)
    
    metrics_text = "\n".join([
        f"MAE: {mae:.2f} {config['unit']}",
        f"RMSE: {np.sqrt(mse):.2f} {config['unit']}",
        f"SSIM: {ssim:.3f}",
        f"Max Value: {np.max(target):.1f} {config['unit']}",
        f"Min Value: {np.min(target):.1f} {config['unit']}"
    ])
    
    axs[4].text(0.5, 0.5, metrics_text, 
                ha='center', va='center', 
                fontsize=14, family='monospace')
    axs[4].axis('off')

    # Add DEM visualization from second channel
    dem = input_seq[1]  # DEM is second channel
    dem_ax = axs[0].inset_axes([0.05, 0.05, 0.3, 0.3])
    dem_ax.imshow(dem, cmap='terrain', alpha=0.8)
    dem_ax.set_title('Elevation (m)')
    dem_ax.axis('on')

    plt.tight_layout()
    return fig


def normalise(ds):
    normalized_ds = ds.copy()
    
    # Normalize Precipitation (Min-Max Scaling)
    if 'pr' in ds:
        precip = ds['pr']
        normalized_ds['pr'] = (precip - precip.min()) / (precip.max() - precip.min())

    if 'tasmean' in ds:
        precip = ds['tasmean']
        normalized_ds['tasmean'] = (precip - precip.min()) / (precip.max() - precip.min())

    if 'ssrd' in ds:
        precip = ds['ssrd']
        normalized_ds['ssrd'] = (precip - precip.min()) / (precip.max() - precip.min())
    
    # Standardize DEM (Z-score Normalization)
    if 'elevation' in ds:
        dem = ds['elevation']
        normalized_ds['elevation'] = (dem - dem.mean()) / dem.std()
    
    return normalized_ds



if __name__ == '__main__':

    cluster = LocalCluster(n_workers=3, memory_limit="39GB")
    client = Client(cluster)

    # initialize timing
    start_time = time.monotonic()
        
    # initialize network filename
    state_file = NAMING_Model.state_file(
        NET, PREDICTAND, ERA5_PREDICTORS, ERA5_PLEVELS, WET_DAY_THRESHOLD, dem=DEM,
        dem_features=DEM_FEATURES, doy=DOY, stratify=STRATIFY)
    
    state_file = MODEL_PATH.joinpath(PREDICTAND, state_file)
    target = TARGET_PATH.joinpath(PREDICTAND)

    # check if output path exists
    if not target.exists():
        target.mkdir(parents=True, exist_ok=True)
    # initialize logging
    log_file = state_file.with_name(state_file.name + "_log.txt")
    
    if log_file.exists():
        log_file.unlink()
    dictConfig(log_conf(log_file))

    # check if target dataset already exists
    target = target.joinpath(state_file.name + '.nc')
    if target.exists() and not OVERWRITE:
        LogConfig.init_log('{} already exists.'.format(target))
        sys.exit()

    LogConfig.init_log('Initializing downscaling for period: {}'.format(
        ' - '.join([str(CALIB_PERIOD[0]), str(CALIB_PERIOD[-1])])))

    # initialize ERA5 predictor dataset
    LogConfig.init_log(f'Initializing ERA5 predictors:{ERA5_PREDICTORS}')
    Era5 = ERA5Dataset(ERA5_PATH.joinpath(''), ERA5_PREDICTORS,
                       plevels=ERA5_PLEVELS)
    Era5_ds = Era5.merge(chunks=CHUNKS)
    Era5_ds = Era5_ds.rename({'lon': 'x','lat': 'y'})
    
    # initialize OBS predictand dataset
    LogConfig.init_log('Initializing observations for predictand: {}'
                       .format(PREDICTAND))

    # read in-situ gridded observations
    Obs_ds = search_files(OBS_PATH.joinpath(PREDICTAND), '.nc$').pop()
    Obs_ds = xr.open_dataset(Obs_ds)
    Obs_ds = Obs_ds.rename({'longitude': 'x','latitude': 'y'})

    # whether to use digital elevation model
    if DEM:
        # digital elevation model: Copernicus EU-Dem v1.1
        dem = search_files(DEM_PATH, '^interTwin_dem_chelsa_stage2.nc$').pop()

        # read elevation and compute slope and aspect
        dem = ERA5Dataset.dem_features(
            dem, {'y': Era5_ds.y, 'x': Era5_ds.x},
            add_coord={'time': Era5_ds.time})

        # check whether to use slope and aspect
        if not DEM_FEATURES:
            dem = dem.drop_vars(['slope', 'aspect']).chunk(Era5_ds.chunks)

        # add dem to set of predictor variables
        dem = dem.chunk(Era5_ds.chunks)
        Era5_ds = xr.merge([Era5_ds, dem])
    
    # initialize training data
    LogConfig.init_log('Initializing training data.')

    # split calibration period into training and validation period
    if PREDICTAND == 'pr' and STRATIFY:
        # stratify training and validation dataset by number of
        # observed wet days for precipitation
        wet_days = (Obs_ds.sel(time=CALIB_PERIOD).mean(dim=('y', 'x'))
                    >= WET_DAY_THRESHOLD).to_array().values.squeeze()
        train, valid = train_test_split(
            CALIB_PERIOD, stratify=wet_days, test_size=VALID_SIZE)

        # sort chronologically
        train, valid = sorted(train), sorted(valid)
        Era5_train, Obs_train = Era5_ds.sel(time=train), Obs_ds.sel(time=train)
        Era5_valid, Obs_valid = Era5_ds.sel(time=valid), Obs_ds.sel(time=valid)
    else:
        LogConfig.init_log('We are not calculating Stratified Precipitation based on Wet Days here!')

    
    Era5_ds = Era5_ds.sel(y=slice(45.985, 47.05), x=slice(10.8, 11.87)).chunk(CHUNKS) # Just cut short this area for using the full domain
    Obs_ds = Obs_ds.sel(y=slice(45.985, 47.05), x=slice(10.8, 11.87)).chunk(CHUNKS)  # Just cut short this area for using the full domain
    Era5_ds = Era5_ds.fillna(0)
    Obs_ds = Obs_ds.fillna(0)
    Era5_ds = Era5_ds.astype(np.float32)

    # training and validation dataset
    Era5_train, Obs_train = Era5_ds.sel(time=CALIB_PERIOD), Obs_ds.sel(time=CALIB_PERIOD)
    Era5_test, Obs_test = Era5_ds.sel(time=TEST_PERIOD), Obs_ds.sel(time=TEST_PERIOD)


    Era5_train = normalise(Era5_train)
    Era5_train = doy_encoding(Era5_train, Obs_train, doy=DOY)

    Era5_test = normalise(Era5_test)
    Era5_test = doy_encoding(Era5_test, Obs_test, doy=DOY)

    LogConfig.init_log(f'Predictors : {Era5_train}')
    LogConfig.init_log(f'Targets : {Obs_train}')

    LogConfig.init_log(f'Predictors_test : {Era5_test}')
    LogConfig.init_log(f'Targets_test : {Obs_test}')
    
    X_train = stacker(Era5_train).compute()
    y_train = stacker(Obs_train).compute()

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    X_test = stacker(Era5_test).compute()
    y_test = stacker(Obs_test).compute()

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    LogConfig.init_log(f'predictors_train shape : {X_train.shape}')
    LogConfig.init_log(f'predictand_train shape : {y_train.shape}')

    LogConfig.init_log(f'predictors_test shape : {X_test.shape}')
    LogConfig.init_log(f'predictand_test shape : {y_test.shape}')

    batch_size = 32

    train_loader = prepare_data(X_train, y_train, batch_size=batch_size)
    test_loader = prepare_data(X_test, y_test, batch_size=batch_size)

    # Get the first batch from the train_loader
    train_features, train_labels = next(iter(train_loader))
    print(f"Train batch shape: {train_features.shape}, Labels shape: {train_labels.shape}")

    # Get the first batch from the test_loader
    test_features, test_labels = next(iter(test_loader))
    print(f"Test batch shape: {test_features.shape}, Labels shape: {test_labels.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 3: Initialize models
    generator = Generator(in_channels=4, out_channels=1)
    discriminator = PatchGANDiscriminator(in_channels=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    #generator.load_state_dict(torch.load("/home/sdhinakaran/consoldiated_downScaleML/downScaleML/downscaleml/main/generator_epoch_46.pth"))
    #discriminator.load_state_dict(torch.load("/home/sdhinakaran/consoldiated_downScaleML/downScaleML/downscaleml/main/discriminator_epoch_46.pth"))
    
    # Step 4: Define optimizers and schedulers
    optimizer_G = optim.Adam(
    generator.parameters(), 
    lr=2e-4,  # Slightly higher learning rate
    betas=(0.9, 0.999),  # Default betas
    weight_decay=1e-4  # Regularization
    )
    
    optimizer_D = optim.Adam(
        discriminator.parameters(), 
        lr=1e-4,  # Keep discriminator learning rate lower
        betas=(0.5, 0.9),  # Keep betas as is for discriminator
        weight_decay=1e-4  # Regularization
    )
    

    scheduler_G = CosineAnnealingLR(
        optimizer_G, 
        T_max=10,  # Cycle length (epochs)
        eta_min=1e-6  # Minimum learning rate
    )
    
    scheduler_D = CosineAnnealingLR(
        optimizer_D, 
        T_max=10,  
        eta_min=1e-6
    )


    # Training parameters
    num_epochs = 500
    patience = 50
    min_delta = 0.001
    save_samples_every = 2
    lambda1=1.0  # Content loss (L1) weight
    lambda2=0.1  # Perceptual loss weight
    lambda3=0.001  # Adversarial loss weight
    lambda4=0.5 # MS-SSIM loss weight
    lambda5=0.2  # Frequency loss weight
    lambda_gp=10.0  
    n_critic = 1
    save_all_every = 5

    
    # Execute training
    train_esrgan(
        dataloader_train=train_loader,
        dataloader_test=test_loader,
        generator=generator,
        discriminator=discriminator,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D,
        scheduler_G=scheduler_G,
        scheduler_D=scheduler_D,
        device=device,
        num_epochs=num_epochs,
        patience=patience,
        min_delta=min_delta,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3,
        lambda4=lambda4,
        lambda5=lambda5,
        lambda_gp=lambda_gp,
        n_critic=n_critic,
        save_samples_every=save_samples_every,
        save_all_every=save_all_every
        )
    
    LogConfig.init_log('Training Completed!!! SMILE PLEASE')


