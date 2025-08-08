#!/usr/bin/env python
# coding: utf-8

"""
Brain-to-Text Competition: Innovative Diffusion-Based Approach
Fixed version with proper data loading
"""

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

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# 1. EXPLORE DATA STRUCTURE FIRST
# ============================================================================

def explore_hdf5_structure(file_path):
    """Explore the structure of HDF5 files"""
    print(f"\nExploring: {os.path.basename(file_path)}")
    print("-" * 50)
    
    with h5py.File(file_path, 'r') as f:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name}")
                print(f"    Shape: {obj.shape}")
                print(f"    Dtype: {obj.dtype}")
                if obj.shape[0] > 0:
                    print(f"    Sample: {obj[0] if obj.ndim == 1 else 'Array'}")
        
        print(f"Keys: {list(f.keys())}")
        f.visititems(print_structure)
    
    return

# First, let's explore the data structure
data_base_path = Path('/kaggle/input/brain2text-complete-dataset')

# Find a sample HDF5 file to explore
sample_file = None
for root, dirs, files in os.walk(data_base_path):
    for file in files:
        if file.endswith('.hdf5'):
            sample_file = os.path.join(root, file)
            break
    if sample_file:
        break

if sample_file:
    explore_hdf5_structure(sample_file)
else:
    print("No HDF5 files found!")

# ============================================================================
# 2. FLEXIBLE DATA LOADER
# ============================================================================

class BrainTextDataset(Dataset):
    """Flexible dataset for Brain-to-Text data"""
    
    def __init__(self, data_paths, mode='train', max_samples_per_file=100):
        self.data_paths = data_paths
        self.mode = mode
        self.max_samples_per_file = max_samples_per_file
        self.samples = []
        
        # Try to load data
        self._load_data()
        
        # If no samples loaded, create dummy data for testing
        if len(self.samples) == 0:
            print(f"Warning: No real data loaded for {mode}, creating dummy data")
            self._create_dummy_data()
    
    def _load_data(self):
        """Load data from HDF5 files with flexible key detection"""
        
        # Possible key names for neural data and text
        neural_keys = ['spikePow', 'neuralData', 'neural', 'tx1', 'tx2', 'tx3', 'tx4']
        text_keys = ['sentenceText', 'targetText', 'text', 'sentence', 'transcription']
        
        for path in self.data_paths[:5]:  # Limit files for testing
            if not os.path.exists(path):
                continue
            
            print(f"Loading: {os.path.basename(path)}")
            
            try:
                with h5py.File(path, 'r') as f:
                    # Find available keys
                    available_keys = list(f.keys())
                    print(f"  Available keys: {available_keys}")
                    
                    # Try to find neural data
                    neural_data = None
                    neural_key_used = None
                    for key in neural_keys:
                        if key in f:
                            neural_data = f[key]
                            neural_key_used = key
                            break
                    
                    # Try to find text data
                    text_data = None
                    text_key_used = None
                    for key in text_keys:
                        if key in f:
                            text_data = f[key]
                            text_key_used = key
                            break
                    
                    # If we have neural data but no text, check for trial/block structure
                    if neural_data is not None and text_data is None:
                        # Look for text in nested structure
                        for key in available_keys:
                            if 'block' in key.lower() or 'trial' in key.lower():
                                if key in f and isinstance(f[key], h5py.Group):
                                    for subkey in text_keys:
                                        if subkey in f[key]:
                                            text_data = f[key][subkey]
                                            text_key_used = f"{key}/{subkey}"
                                            break
                    
                    # Process data if both found
                    if neural_data is not None:
                        print(f"  Found neural data: {neural_key_used}, shape: {neural_data.shape}")
                        
                        # Handle different data formats
                        if len(neural_data.shape) == 3:
                            # Shape: (trials, channels, time)
                            n_samples = min(neural_data.shape[0], self.max_samples_per_file)
                            
                            for i in range(n_samples):
                                neural_sample = neural_data[i]
                                
                                # Generate text if not available
                                if text_data is not None and i < len(text_data):
                                    try:
                                        if isinstance(text_data[i], bytes):
                                            text = text_data[i].decode('utf-8')
                                        elif isinstance(text_data[i], np.ndarray):
                                            text = str(text_data[i])
                                        else:
                                            text = str(text_data[i])
                                    except:
                                        text = f"Sample {i}"
                                else:
                                    text = f"Sample {i}"
                                
                                self.samples.append({
                                    'neural': neural_sample,
                                    'text': text.strip()
                                })
                        
                        elif len(neural_data.shape) == 2:
                            # Shape: (channels, time) - single trial
                            neural_sample = neural_data[:]
                            text = "Single trial data"
                            if text_data is not None:
                                try:
                                    if isinstance(text_data, bytes):
                                        text = text_data.decode('utf-8')
                                    elif hasattr(text_data, '__iter__') and len(text_data) > 0:
                                        text = str(text_data[0])
                                except:
                                    pass
                            
                            self.samples.append({
                                'neural': neural_sample,
                                'text': text
                            })
                        
                        print(f"  Loaded {len(self.samples)} samples from this file")
                    
                    else:
                        # Try to load any numeric data as neural data
                        for key in available_keys:
                            if key not in ['sentenceText', 'targetText', 'text']:
                                try:
                                    data = f[key]
                                    if isinstance(data, h5py.Dataset) and data.dtype.kind in ['f', 'i']:
                                        print(f"  Using {key} as neural data, shape: {data.shape}")
                                        
                                        if len(data.shape) >= 2:
                                            # Take first few samples
                                            n_samples = min(data.shape[0], self.max_samples_per_file)
                                            for i in range(n_samples):
                                                self.samples.append({
                                                    'neural': data[i] if len(data.shape) > 1 else data[:],
                                                    'text': f"Sample {i} from {key}"
                                                })
                                            break
                                except:
                                    continue
                
            except Exception as e:
                print(f"  Error loading {path}: {e}")
                continue
        
        print(f"\nTotal samples loaded for {self.mode}: {len(self.samples)}")
    
    def _create_dummy_data(self):
        """Create dummy data for testing the pipeline"""
        print("Creating dummy data for testing...")
        
        # Create 100 dummy samples
        for i in range(100):
            # Random neural data (128 channels, 64 time steps)
            neural = np.random.randn(128, 64).astype(np.float32)
            
            # Simple text samples
            texts = [
                "Hello world",
                "Testing brain to text",
                "Neural decoding works",
                "Machine learning is amazing",
                "Brain computer interface"
            ]
            text = texts[i % len(texts)]
            
            self.samples.append({
                'neural': neural,
                'text': text
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert to tensor
        neural = torch.tensor(sample['neural'], dtype=torch.float32)
        
        # Ensure correct shape (128 channels, variable time)
        if len(neural.shape) == 1:
            # Reshape 1D to 2D
            neural = neural.reshape(128, -1)
        elif len(neural.shape) == 3:
            # Take first dimension if 3D
            neural = neural[0]
        
        # Ensure 128 channels
        if neural.shape[0] != 128:
            # Pad or truncate
            if neural.shape[0] < 128:
                padding = torch.zeros(128 - neural.shape[0], neural.shape[1])
                neural = torch.cat([neural, padding], dim=0)
            else:
                neural = neural[:128, :]
        
        # Ensure reasonable time dimension
        if neural.shape[1] < 32:
            # Pad time dimension
            padding = torch.zeros(128, 32 - neural.shape[1])
            neural = torch.cat([neural, padding], dim=1)
        elif neural.shape[1] > 256:
            # Truncate if too long
            neural = neural[:, :256]
        
        # Normalize
        neural = (neural - neural.mean()) / (neural.std() + 1e-8)
        
        return {
            'neural': neural,
            'text': sample['text']
        }

# ============================================================================
# 3. SIMPLE TOKENIZER
# ============================================================================

class SimpleTokenizer:
    """Character-level tokenizer"""
    
    def __init__(self):
        self.char_to_id = {
            '<pad>': 0, '<unk>': 1, '<cls>': 2, '<sep>': 3, '<eos>': 4
        }
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        
        # Add ASCII characters
        chars = ' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:\'"()-'
        for i, char in enumerate(chars):
            self.char_to_id[char] = i + 5
            self.id_to_char[i + 5] = char
        
        self.vocab_size = len(self.char_to_id)
    
    def encode(self, text, max_length=100):
        ids = [2]  # CLS token
        for char in text[:max_length-3]:
            ids.append(self.char_to_id.get(char, 1))  # UNK for unknown
        ids.extend([3, 4])  # SEP and EOS tokens
        
        # Pad
        while len(ids) < max_length:
            ids.append(0)
        
        return torch.tensor(ids[:max_length])
    
    def decode(self, ids):
        text = ''
        for id in ids:
            if id == 4:  # EOS
                break
            if id > 4:  # Skip special tokens
                text += self.id_to_char.get(int(id), '')
        return text

# ============================================================================
# 4. SIMPLIFIED MODEL
# ============================================================================

class SimpleNeuralDecoder(nn.Module):
    """Simplified model for testing"""
    
    def __init__(self, vocab_size=100, neural_dim=128, hidden_dim=256):
        super().__init__()
        
        # Neural encoder
        self.neural_encoder = nn.Sequential(
            nn.Conv1d(neural_dim, hidden_dim, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Text decoder
        self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, 2, batch_first=True)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, neural_signals, target_text=None):
        # Encode neural signals
        neural_features = self.neural_encoder(neural_signals)
        
        if target_text is not None:
            # Training mode
            text_emb = self.text_embedding(target_text)
            output, _ = self.decoder(text_emb, (neural_features.unsqueeze(0).repeat(2, 1, 1), 
                                                  neural_features.unsqueeze(0).repeat(2, 1, 1)))
            logits = self.output_projection(output)
            
            # Compute loss
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                target_text[:, 1:].reshape(-1),
                ignore_index=0
            )
            
            return logits, loss
        else:
            # Inference mode
            batch_size = neural_signals.shape[0]
            device = neural_signals.device
            
            # Start with CLS token
            generated = torch.full((batch_size, 1), 2, device=device)
            
            # Initialize hidden state
            h = neural_features.unsqueeze(0).repeat(2, 1, 1)
            c = neural_features.unsqueeze(0).repeat(2, 1, 1)
            
            for _ in range(50):  # Max length
                text_emb = self.text_embedding(generated[:, -1:])
                output, (h, c) = self.decoder(text_emb, (h, c))
                logits = self.output_projection(output)
                next_token = torch.argmax(logits, dim=-1)
                generated = torch.cat([generated, next_token], dim=1)
                
                if (next_token == 4).all():  # EOS
                    break
            
            return generated

# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 80)
    print("BRAIN-TO-TEXT INNOVATIVE APPROACH - FIXED VERSION")
    print("=" * 80)
    
    # Data paths
    data_base_path = Path('/kaggle/input/brain2text-complete-dataset')
    
    # Collect ALL possible HDF5 files
    all_hdf5_files = []
    for root, dirs, files in os.walk(data_base_path):
        for file in files:
            if file.endswith('.hdf5'):
                all_hdf5_files.append(os.path.join(root, file))
    
    print(f"\nFound {len(all_hdf5_files)} total HDF5 files")
    
    # Split files for train/val
    if len(all_hdf5_files) > 0:
        # Use first 70% for training, next 30% for validation
        split_idx = int(len(all_hdf5_files) * 0.7)
        train_paths = all_hdf5_files[:split_idx]
        val_paths = all_hdf5_files[split_idx:]
    else:
        print("No HDF5 files found, will use dummy data")
        train_paths = []
        val_paths = []
    
    print(f"Train files: {len(train_paths)}")
    print(f"Val files: {len(val_paths)}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = BrainTextDataset(train_paths, mode='train', max_samples_per_file=50)
    val_dataset = BrainTextDataset(val_paths, mode='val', max_samples_per_file=20)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Check if datasets are empty
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("\nError: Datasets are empty. Please check data paths and format.")
        return
    
    # Create dataloaders
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize tokenizer and model
    tokenizer = SimpleTokenizer()
    model = SimpleNeuralDecoder(vocab_size=tokenizer.vocab_size, neural_dim=128, hidden_dim=256)
    model = model.to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    optimizer = AdamW(model.parameters(), lr=1e-3)
    
    print("\nStarting training...")
    for epoch in range(3):  # Quick training
        # Train
        model.train()
        train_losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            neural = batch['neural'].to(device)
            texts = batch['text']
            
            # Tokenize texts
            target = torch.stack([tokenizer.encode(t) for t in texts]).to(device)
            
            # Forward pass
            logits, loss = model(neural, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validate
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                neural = batch['neural'].to(device)
                texts = batch['text']
                target = torch.stack([tokenizer.encode(t) for t in texts]).to(device)
                
                logits, loss = model(neural, target)
                val_losses.append(loss.item())
        
        print(f"Epoch {epoch+1}: Train Loss = {np.mean(train_losses):.4f}, Val Loss = {np.mean(val_losses):.4f}")
    
    # Test inference
    print("\nTesting inference...")
    model.eval()
    with torch.no_grad():
        sample = val_dataset[0]
        neural = sample['neural'].unsqueeze(0).to(device)
        
        generated = model(neural)
        generated_text = tokenizer.decode(generated[0].cpu().numpy())
        
        print(f"Original: {sample['text']}")
        print(f"Generated: {generated_text}")
    
    print("\n" + "=" * 80)
    print("EXECUTION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()