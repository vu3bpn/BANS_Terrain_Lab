#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 15:37:51 2026

@author: bipin
"""
from scipy import spatial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
from sklearn.neighbors import NearestNeighbors

import math
import geopandas
import pandas
import os
import random
import numpy as np
from functools import lru_cache
import ipdb
from config import *
from misc_utilities import *


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, V)

    def split_heads(self, x):
        # x: (batch, seq_len, d_model) → (batch, n_heads, seq_len, d_k)
        batch, seq_len, _ = x.size()
        x = x.view(batch, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, x, mask=None):
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))

        attn = self.scaled_dot_product_attention(Q, K, V, mask)

        # Merge heads back: (batch, n_heads, seq_len, d_k) → (batch, seq_len, d_model)
        batch, _, seq_len, _ = attn.size()
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.W_o(attn)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class PatchNormaliser(nn.Module):
    """
    Normalises each channel of a patch independently across all points,
    then denormalises using the saved statistics.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.mu = 1
        self.sigma = 1
        

    def normalise(self, x: torch.Tensor):
        """
        x: (B, N, F)
        Returns normalised x, and the statistics needed to undo it.
        μ, σ: (B, 1, F)  — one value per sample per feature channel
        """
        self.mu    = x.mean(dim=1, keepdim=True)   # mean over N points
        self.sigma = x.std(dim=1, keepdim=True) + self.eps
        #breakpoint()
        #self.mu[3:] = np.mean(self.mu[3:])
        #self.sigma[3:] = np.mean(self.sigma[3:]) 
        #rgbi_mu = self.mu[:,:,3:].mean(axis=2)
        #rgbi_sigma = self.sigma[:,:,3:].max(axis=2)
        #rgb1_sigma = np.max(self.sigma[3:])
        #self.mu[3:] = rgbi_mu
        #self.sigma[3:] = rgb1_sigma
        self.mu[:,:,3:] = self.mu[:,:,3:].mean()
        self.sigma[:,:,3:] = self.sigma[:,:,3:].max()
        

        x_norm = (x - self.mu) / self.sigma
        return x_norm 

    def denormalise(self, x_norm):
        """Reverse the normalisation using saved μ and σ."""
        return x_norm * self.sigma + self.mu

    def forward(self, x):
        """
        Convenience: normalise only (use denormalise() separately after model).
        Returns (x_norm, mu, sigma).
        """
        return self.normalise(x)

class dummy_model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x




class Transformer7D(nn.Module):
    """
    A simple transformer that takes vectors of length 7
    and generates vectors of length 7.

    Args:
        input_dim:  Size of input/output vectors (7).
        d_model:    Internal model dimension.
        n_heads:    Number of attention heads (must divide d_model).
        n_layers:   Number of stacked transformer blocks.
        d_ff:       Feed-forward hidden dimension.
        dropout:    Dropout probability.
    """
    def __init__(
        self,
        input_dim: int = 7,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()


        self.norm = PatchNormaliser(eps=1e-8)

        # Project input_dim → d_model
        self.embedding = nn.Linear(input_dim, d_model)

        # Stack of transformer blocks
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        # Project d_model → input_dim (back to 7)
        self.output_projection = nn.Linear(d_model, input_dim)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        Args:
            x:    Tensor of shape (batch_size, seq_len, 7)
            mask: Optional attention mask

        Returns:
            Tensor of shape (batch_size, seq_len, 7)
        """
        x  = self.norm(x)  # normalise input
        x = self.embedding(x)           # (batch, seq_len, d_model)
        for layer in self.layers:
            x = layer(x, mask)          # (batch, seq_len, d_model)
        x = self.output_projection(x)  # (batch, seq_len, 7)
        x = self.norm.denormalise(x)  # denormalise output
        return x



# ── 2. Training loop ──────────────────────────────────────────────────────────

def train(
    model,
    train_loader,
    val_loader,
    n_epochs=2,
    lr=1e-3,
    device="cpu",
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    #criterion = nn.MSELoss()
    #criterion = nn.SmoothL1Loss(beta=1.0)
    #criterion = nn.HuberLoss(delta=1.0)
    criterion = nn.L1Loss()

    history_dict = {"train_loss": [], "val_loss": []}

    for epoch in range(1, n_epochs + 1):
        # ── train ──
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()          # clear old gradients
            pred = model(x_batch)          # forward pass → (batch, seq, 7)
            loss = criterion(pred, y_batch)  # compute MSE
            loss.backward()                # backprop
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()               # update weights
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # ── validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                try:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    pred = model(x_batch)
                    val_loss += criterion(pred, y_batch).item()
                except Exception as e:   
                    log(f"Error in validating data {x_batch.shape} {y_batch.shape}")
                    print(f"{e}")
                    
        val_loss /= len(val_loader)

        scheduler.step()

        history_dict["train_loss"].append(train_loss)
        history_dict["val_loss"].append(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{n_epochs}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    return history_dict


# ── 3. Save & load helpers ────────────────────────────────────────────────────

def save_checkpoint(model, path="checkpoint.pt"):
    torch.save(model.state_dict(), path)
    print(f"Saved to {path}")

def load_checkpoint(model, path="checkpoint.pt"):
    model.load_state_dict(torch.load(path, map_location="cpu"))
    print(f"Loaded from {path}")
    return model






#-----------data----------------

def align_by_nearest(df1,df2,x="X",y="Y"):
    coords1 = df1[[x,y]].to_numpy(dtype=float)
    coords2 = df2[[x,y]].to_numpy(dtype=float)
    tree = spatial.cKDTree(coords2)
    distances, indices = tree.query(coords1, k=1)
    df2_matched = (
        df2.iloc[indices]
           .reset_index(drop=True)
    )
    return df2_matched

class StreamingPointCloudDataset(IterableDataset):
    def __init__(self,batch_size = 64,n_batches = 32,seq_len = 1024):
        super().__init__()
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.seq_len = seq_len
        self.files_list = []
        self.selected_cols = ["X","Y","Z","red","green","blue","intensity"]    
        self.read_files()


    def read_files(self):
        data_info_df = pandas.read_csv(data_info_file)
        filename_paths = {os.path.split(x)[-1]:x for x in data_info_df['filename']}
        for las_file in las_vect_dict:
            vect_file = las_vect_dict[las_file]
            shapefile_path = os.path.join(vector_dir,vect_file)
            las_file_path = filename_paths[las_file]
            las_file_name = os.path.splitext(las_file)[0]
            dtm_file = las_file_name+"_dtm.las"
            dtm_file_path = os.path.join(dtm_dir,dtm_file)
            gdf = geopandas.read_file(shapefile_path)
            for row1 in gdf.iterrows():
                id = row1[1]['id']
                geometry = row1[1].geometry
                self.files_list.append((las_file_path,dtm_file_path,geometry))

    def __len__(self):
        return self.n_batches
    
    @lru_cache(maxsize=5)
    def get_df_tree(self,file_tuple):
        las_file_path, dtm_file_path, geometry = file_tuple
        dtm_record,dtm_header = subset_with_geom(dtm_file_path,geometry)
        dsm_record,dsm_header = subset_with_geom(las_file_path,geometry)
        dtm_df = pandas.DataFrame(dtm_record.array)
        dsm_df = pandas.DataFrame(dsm_record.array)
        dsm_df = align_by_nearest(dtm_df,dsm_df)
        xy = dtm_df[["X","Y"]]        
        tree = spatial.cKDTree(xy)
        return dtm_df,dsm_df,tree

    def get_dataset(self):
        file_tuple = random.choice(self.files_list)
        dtm_df,dsm_df,tree = self.get_df_tree(file_tuple)
        
        center_idx = random.sample(range(len(dsm_df)),1)[0]
        center_point = dsm_df.iloc[center_idx][["X","Y"]]
        dist, tree_idx = tree.query(center_point, k=self.seq_len)
        
        dsm_data = dsm_df.iloc[tree_idx]
        dtm_data =  dtm_df.iloc[tree_idx]
           
        input_dataset = np.array(dsm_data[self.selected_cols])
        target_dataset = np.array(dtm_data[self.selected_cols])
        return torch.tensor(input_dataset,dtype=torch.float32),torch.tensor(target_dataset,dtype=torch.float32)
       
        

    def __iter__(self):
        batches = range(self.n_batches)  
        for batch in batches:
            dataset = self.get_dataset()
            #print(dataset.shape)
            yield dataset





    
#%% Testing    
if __name__ == "__main1__":
    train_set = StreamingPointCloudDataset()
    for ds1 in train_set:
        x,y = ds1
        print(x.shape,y.shape)
    
    
#--------------model---------------
if __name__ == "__main__":
    model = Transformer7D(
        input_dim=7,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
        )
    



#%% ─── Training ─────────────────────────────────────────────────────────────
if __name__ == "__main__":  
    #DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE     = "cpu"
    BATCH_SIZE = 64
    N_EPOCHS   = 1000
    INPUT_DIM  = 7
    log(f"Device: {DEVICE}")
    
    '''
    n_val = int(0.2*len(dataset))
    train_set, val_set = torch.utils.data.random_split(
        dataset, [len(dataset) - n_val, n_val]
    )
    '''
    train_set = StreamingPointCloudDataset()
    val_set = StreamingPointCloudDataset()
    
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE)
    print("Training")

    
    if os.path.exists(dtm_model_path):
        load_checkpoint(model, dtm_model_path)
    
    history_dict = train(model, train_loader, val_loader,
                    n_epochs=N_EPOCHS, lr=1e-1, device=DEVICE)

    # Save
    save_checkpoint(model,dtm_model_path)

    


# ── Quick demo ────────────────────────────────────────────────────────────────
if __name__ == "__main1__":  
    model.eval()
    with torch.no_grad():
        output = model(x)

   
