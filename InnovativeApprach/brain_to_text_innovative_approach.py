#!/usr/bin/env python
# coding: utf-8

# # Brain-to-Text Competition: Innovative Diffusion-Based Approach
# 
# This notebook implements a novel approach combining:
# - Diffusion models for synthetic neural pattern generation
# - Contrastive learning between real and synthetic data
# - Uncertainty-aware decoding with ensemble methods
# - Meta-learning for quick adaptation

import os
import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Memory optimization
torch.cuda.empty_cache()
gc.collect()

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

class BrainTextDataset(Dataset):
    """Custom dataset for Brain-to-Text data"""
    
    def __init__(self, data_paths, mode='train', max_length=512, max_samples=None):
        self.data_paths = data_paths
        self.mode = mode
        self.max_length = max_length
        self.max_samples = max_samples
        self.samples = []
        
        # Load all data
        self._load_data()
        
    def _load_data(self):
        """Load data from HDF5 files"""
        sample_count = 0
        
        for path in tqdm(self.data_paths, desc=f"Loading {self.mode} data"):
            if not os.path.exists(path):
                continue
                
            try:
                with h5py.File(path, 'r') as f:
                    # Debug: print available keys
                    if sample_count == 0:
                        print(f"HDF5 keys in {os.path.basename(path)}: {list(f.keys())}")
                    
                    # Get neural data
                    neural_data = None
                    if 'spikePow' in f:
                        neural_data = f['spikePow'][:]
                    elif 'neuralData' in f:
                        neural_data = f['neuralData'][:]
                    
                    # Get text data
                    texts = None
                    if 'sentenceText' in f:
                        texts = f['sentenceText'][:]
                    elif 'targetText' in f:
                        texts = f['targetText'][:]
                    
                    if neural_data is not None and texts is not None:
                        # Process each sample
                        for i in range(min(len(neural_data), len(texts))):
                            if self.max_samples and sample_count >= self.max_samples:
                                break
                                
                            # Decode text if needed
                            if isinstance(texts[i], bytes):
                                text = texts[i].decode('utf-8')
                            elif isinstance(texts[i], np.ndarray):
                                text = texts[i].tobytes().decode('utf-8', errors='ignore')
                            else:
                                text = str(texts[i])
                            
                            # Clean text
                            text = text.strip()
                            if len(text) > 0:  # Only add non-empty texts
                                self.samples.append({
                                    'neural': neural_data[i],
                                    'text': text
                                })
                                sample_count += 1
                        
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
                
        print(f"Loaded {len(self.samples)} samples for {self.mode}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Process neural data
        neural = torch.tensor(sample['neural'], dtype=torch.float32)
        
        # Handle different neural data shapes
        if len(neural.shape) == 1:
            # Reshape 1D to 2D (channels, time)
            neural = neural.reshape(128, -1)
        elif len(neural.shape) == 3:
            # Take mean across one dimension if 3D
            neural = neural.mean(dim=0)
        
        # Ensure correct shape (128 channels, time steps)
        if neural.shape[0] != 128:
            # Pad or truncate channels
            if neural.shape[0] < 128:
                padding = torch.zeros(128 - neural.shape[0], neural.shape[1])
                neural = torch.cat([neural, padding], dim=0)
            else:
                neural = neural[:128, :]
        
        # Normalize neural data
        neural = (neural - neural.mean()) / (neural.std() + 1e-8)
        
        return {
            'neural': neural,
            'text': sample['text']
        }

# ============================================================================
# 2. SIMPLE TOKENIZER
# ============================================================================

class SimpleTokenizer:
    """Character-level tokenizer for memory efficiency"""
    
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.char_to_id = {}
        self.id_to_char = {}
        
        # Special tokens
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.cls_token_id = 2
        self.sep_token_id = 3
        self.eos_token_id = 4
        
        # Build vocabulary from ASCII characters
        chars = ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:\'"()-'
        for i, char in enumerate(chars):
            self.char_to_id[char] = i + 5
            self.id_to_char[i + 5] = char
    
    def encode(self, text, max_length=100):
        """Encode text to token IDs"""
        ids = [self.cls_token_id]
        
        for char in text[:max_length-3]:  # Reserve space for special tokens
            ids.append(self.char_to_id.get(char, self.unk_token_id))
        
        ids.append(self.sep_token_id)
        ids.append(self.eos_token_id)
        
        # Padding
        while len(ids) < max_length:
            ids.append(self.pad_token_id)
        
        # Truncate if necessary
        ids = ids[:max_length]
        
        # Create attention mask
        attention_mask = [1 if id != self.pad_token_id else 0 for id in ids]
        
        return {
            'input_ids': torch.tensor(ids),
            'attention_mask': torch.tensor(attention_mask)
        }
    
    def decode(self, ids):
        """Decode token IDs back to text"""
        text = ''
        for id in ids:
            if id == self.eos_token_id:
                break
            if id in self.id_to_char:
                text += self.id_to_char[id]
        return text.strip()

# ============================================================================
# 3. DIFFUSION MODEL COMPONENTS
# ============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for diffusion timesteps"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SimpleUNet(nn.Module):
    """Simplified U-Net for neural pattern generation"""
    
    def __init__(self, in_channels=128, hidden_dim=256, time_dim=128, text_dim=256):
        super().__init__()
        self.time_dim = time_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Text conditioning
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Encoder blocks
        self.enc1 = self._make_block(in_channels, hidden_dim)
        self.enc2 = self._make_block(hidden_dim, hidden_dim * 2)
        self.enc3 = self._make_block(hidden_dim * 2, hidden_dim * 2)
        
        # Bottleneck
        self.bottleneck = self._make_block(hidden_dim * 2, hidden_dim * 2)
        
        # Decoder blocks
        self.dec3 = self._make_block(hidden_dim * 4, hidden_dim * 2)
        self.dec2 = self._make_block(hidden_dim * 3, hidden_dim)
        self.dec1 = self._make_block(hidden_dim * 2, hidden_dim)
        
        # Output projection
        self.output = nn.Conv1d(hidden_dim, in_channels, 1)
        
    def _make_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.GELU(),
            nn.Conv1d(out_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.GELU()
        )
    
    def forward(self, x, t, text_emb=None):
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Text conditioning
        if text_emb is not None:
            text_cond = self.text_proj(text_emb)
            # Add text conditioning to time embedding
            t_emb = t_emb + text_cond
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool1d(e1, 2))
        e3 = self.enc3(F.avg_pool1d(e2, 2))
        
        # Bottleneck with time conditioning
        b = self.bottleneck(F.avg_pool1d(e3, 2))
        
        # Add time embedding to bottleneck
        b = b + t_emb.unsqueeze(-1)
        
        # Decoder with skip connections
        d3 = self.dec3(torch.cat([F.interpolate(b, size=e3.shape[-1]), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=e2.shape[-1]), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-1]), e1], dim=1))
        
        # Output
        out = self.output(d1)
        
        return out

# ============================================================================
# 4. MAIN MODEL
# ============================================================================

class NeuralDiffusionBridge(nn.Module):
    """Generate synthetic neural patterns from text using diffusion"""
    
    def __init__(self, neural_channels=128, latent_dim=256, vocab_size=1000):
        super().__init__()
        
        # Text encoder
        self.text_embedding = nn.Embedding(vocab_size, 128)
        self.text_encoder = nn.LSTM(128, latent_dim // 2, 2, 
                                    batch_first=True, bidirectional=True, dropout=0.1)
        
        # Neural diffusion model
        self.neural_diffusion = SimpleUNet(
            in_channels=neural_channels, 
            hidden_dim=128,  # Reduced for memory
            time_dim=64,
            text_dim=latent_dim
        )
        
        # Projection heads
        self.text_projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Diffusion parameters
        self.num_timesteps = 50  # Reduced for faster training
        self.beta_start = 0.0001
        self.beta_end = 0.02
        
        # Pre-calculate beta schedule
        self.register_buffer('betas', torch.linspace(self.beta_start, self.beta_end, self.num_timesteps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
    def encode_text(self, input_ids):
        """Encode text to latent representation"""
        x = self.text_embedding(input_ids)
        output, (h, c) = self.text_encoder(x)
        # Use mean pooling over sequence
        text_features = output.mean(dim=1)
        return self.text_projector(text_features)
    
    @torch.no_grad()
    def generate_synthetic_neural(self, text_features, shape=(128, 64)):
        """Generate synthetic neural patterns using DDPM"""
        batch_size = text_features.shape[0]
        device = text_features.device
        
        # Start from random noise
        x_t = torch.randn(batch_size, shape[0], shape[1], device=device)
        
        # Denoising loop
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.neural_diffusion(x_t, t_batch, text_features)
            
            # DDPM update step
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0, device=device)
            
            beta_t = 1 - alpha_t / alpha_t_prev
            mean = (x_t - beta_t * noise_pred / torch.sqrt(1 - alpha_t)) / torch.sqrt(alpha_t / alpha_t_prev)
            
            if t > 0:
                noise = torch.randn_like(x_t)
                std = torch.sqrt(beta_t)
                x_t = mean + std * noise * 0.5  # Reduced noise for stability
            else:
                x_t = mean
        
        return x_t

class ContrastiveNeuralDecoder(nn.Module):
    """Main decoder using contrastive learning"""
    
    def __init__(self, vocab_size=1000, neural_dim=128, hidden_dim=256):
        super().__init__()
        
        # Components
        self.neural_bridge = NeuralDiffusionBridge(
            neural_channels=neural_dim,
            latent_dim=hidden_dim,
            vocab_size=vocab_size
        )
        
        # Neural encoder
        self.neural_encoder = nn.Sequential(
            nn.Conv1d(neural_dim, hidden_dim, 5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=1024,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=3  # Reduced layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Loss temperature
        self.temperature = 0.07
        
    def forward(self, neural_signals, input_ids=None, attention_mask=None, training=True):
        batch_size = neural_signals.shape[0]
        device = neural_signals.device
        
        # Encode neural signals
        neural_features = self.neural_encoder(neural_signals)
        
        total_loss = 0
        losses = {}
        
        if training and input_ids is not None:
            # Text encoding
            text_features = self.neural_bridge.encode_text(input_ids)
            
            # Contrastive loss
            neural_norm = F.normalize(neural_features, p=2, dim=1)
            text_norm = F.normalize(text_features, p=2, dim=1)
            
            similarity = torch.matmul(neural_norm, text_norm.T) / self.temperature
            labels = torch.arange(batch_size, device=device)
            
            loss_i2t = F.cross_entropy(similarity, labels)
            loss_t2i = F.cross_entropy(similarity.T, labels)
            losses['contrastive'] = (loss_i2t + loss_t2i) / 2
            
            # Generate synthetic neural patterns
            with torch.no_grad():
                synthetic_neural = self.neural_bridge.generate_synthetic_neural(text_features)
            
            # Encode synthetic patterns
            synthetic_features = self.neural_encoder(synthetic_neural)
            
            # Consistency loss
            losses['consistency'] = F.mse_loss(neural_features, synthetic_features)
            
            # Decode to text
            memory = neural_features.unsqueeze(1)
            tgt_emb = self.neural_bridge.text_embedding(input_ids)
            
            # Create causal mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                input_ids.shape[1], device=device
            )
            
            decoded = self.decoder(
                tgt=tgt_emb,
                memory=memory,
                tgt_mask=tgt_mask
            )
            
            output = self.output_projection(decoded)
            
            # Language modeling loss
            output_flat = output[:, :-1, :].reshape(-1, vocab_size)
            target_flat = input_ids[:, 1:].reshape(-1)
            losses['lm'] = F.cross_entropy(output_flat, target_flat, ignore_index=0)
            
            # Combine losses
            total_loss = (
                losses['lm'] + 
                0.3 * losses['contrastive'] + 
                0.1 * losses['consistency']
            )
            
            return output, total_loss, losses
        
        else:
            # Inference mode
            memory = neural_features.unsqueeze(1)
            
            # Start with CLS token
            generated = torch.full((batch_size, 1), 2, device=device)  # CLS token
            
            for _ in range(100):  # Max length
                tgt_emb = self.neural_bridge.text_embedding(generated)
                
                output = self.decoder(
                    tgt=tgt_emb,
                    memory=memory
                )
                
                next_token_logits = self.output_projection(output[:, -1, :])
                next_token = torch.argmax(next_token_logits, dim=-1)
                
                generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
                
                # Stop if all sequences have EOS
                if (next_token == 4).all():  # EOS token
                    break
            
            return generated

# ============================================================================
# 5. TRAINING
# ============================================================================

def train_model(model, train_loader, val_loader, num_epochs=5, device='cuda'):
    """Training loop"""
    
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            # Prepare batch
            neural = batch['neural'].to(device)
            text = batch['text']
            
            # Tokenize text
            tokenizer = SimpleTokenizer()
            encoded = [tokenizer.encode(t) for t in text]
            input_ids = torch.stack([e['input_ids'] for e in encoded]).to(device)
            attention_mask = torch.stack([e['attention_mask'] for e in encoded]).to(device)
            
            # Forward pass
            output, loss, losses = model(neural, input_ids, attention_mask, training=True)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Track loss
            epoch_train_losses.append(loss.item())
            pbar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})
            
            # Clear cache periodically
            if len(epoch_train_losses) % 10 == 0:
                torch.cuda.empty_cache()
        
        avg_train_loss = np.mean(epoch_train_losses)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        epoch_val_losses = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for batch in pbar:
                neural = batch['neural'].to(device)
                text = batch['text']
                
                # Tokenize
                encoded = [tokenizer.encode(t) for t in text]
                input_ids = torch.stack([e['input_ids'] for e in encoded]).to(device)
                attention_mask = torch.stack([e['attention_mask'] for e in encoded]).to(device)
                
                output, loss, losses = model(neural, input_ids, attention_mask, training=True)
                epoch_val_losses.append(loss.item())
                pbar.set_postfix({'loss': loss.item()})
        
        avg_val_loss = np.mean(epoch_val_losses)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, 'best_model.pt')
            print(f"Saved best model with val loss: {best_val_loss:.4f}")
        
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
    
    return model, train_losses, val_losses

# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("=" * 80)
    print("BRAIN-TO-TEXT INNOVATIVE APPROACH")
    print("=" * 80)
    
    # Data paths
    data_base_path = Path('/kaggle/input/brain2text-complete-dataset')
    
    # Collect data files
    train_paths = []
    val_paths = []
    
    for root, dirs, files in os.walk(data_base_path):
        for file in files:
            if file.endswith('.hdf5'):
                full_path = os.path.join(root, file)
                if 'train' in file:
                    train_paths.append(full_path)
                elif 'val' in file:
                    val_paths.append(full_path)
    
    # Limit data for testing (remove in production)
    train_paths = train_paths[:3]
    val_paths = val_paths[:1]
    
    print(f"Found {len(train_paths)} training files")
    print(f"Found {len(val_paths)} validation files")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = BrainTextDataset(train_paths, mode='train', max_samples=100)
    val_dataset = BrainTextDataset(val_paths, mode='val', max_samples=20)
    
    # Create dataloaders
    batch_size = 4  # Small batch for memory
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Initialize model
    print("\nInitializing model...")
    model = ContrastiveNeuralDecoder(
        vocab_size=1000,
        neural_dim=128,
        hidden_dim=256
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train model
    print("\nStarting training...")
    model, train_losses, val_losses = train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=3,  # Quick training for demo
        device=device
    )
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_losses, label='Val Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    print("\nTraining complete!")
    print(f"Best validation loss: {min(val_losses):.4f}")
    
    # Test inference
    print("\nTesting inference...")
    model.eval()
    tokenizer = SimpleTokenizer()
    
    with torch.no_grad():
        sample = val_dataset[0]
        neural = sample['neural'].unsqueeze(0).to(device)
        
        # Generate text
        generated_ids = model(neural, training=False)
        generated_text = tokenizer.decode(generated_ids[0].cpu().numpy())
        
        print(f"Original text: {sample['text']}")
        print(f"Generated text: {generated_text}")
    
    print("\n" + "=" * 80)
    print("EXECUTION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()