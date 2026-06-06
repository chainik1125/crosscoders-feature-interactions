"""
In this file I'll try to implement ``sparse-k" by just training a probe to do reconstruct MLP post-activations
and measuring share of loss recovered
Part 1. Show that a probe can predict a neuron's PRE-activation well for given topk features.
Part 2. Show that a probe can predict a neuron's POST-activation well for given topk features.
#Sanity check - how well can you reconstruct the MLP post-activation from the exact pre-activations? If you
#don't pass this sanity check no point of asking about the topK.
Part 3. Use the reconstructed post-activations to measure the loss recovered as a percentage of the initial 
"""

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors
from IPython.display import HTML
from textwrap import dedent
import tempfile
import pathlib
import webbrowser
from datetime import datetime
import os,sys
import pickle
import time
from tqdm import tqdm
import einops
from datasets import load_dataset
from sleepers.scripts.utils import load_crosscoder_from_wandb
from sleepers.scripts.llms import build_llm_lora
from sleepers.analysis.ft_analysis_util import display_feature_activation_visualization
from sleepers.analysis.analysis_utils import (
    save_dict, 
    load_dict, 
    get_preacts_mlp, 
    get_activations, 
    feature_interactions_mlp, 
    get_preacts_nocontract_faster,
    get_preacts_nocontract,
    feature_interactions_sum,
    feature_interactions_alltokens,
    propagate_preacts
)

from typing import List
from torch.utils.data import DataLoader, Dataset



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)


class SingleNeuronProbe(nn.Module):
    """
    A simple linear probe for reconstructing a single neuron's pre-activation
    from k selected crosscoder features.
    """
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        # Linear layer: [1, k] -> [1]
        self.linear = nn.Linear(k, 1, bias=True)
        
    def forward(self, x):
        """
        Args:
            x: tensor of shape [batch_size, k] containing k feature activations
        Returns:
            tensor of shape [batch_size, 1] containing predicted neuron activation
        """
        return self.linear(x)


class NeuronDataset(Dataset):
    """
    Dataset class for single neuron probing data.
    """
    def __init__(self, input_tensor, label_tensor):
        """
        Args:
            input_tensor: tensor of shape [batch_size, k] containing feature activations
            label_tensor: tensor of shape [batch_size] containing target neuron activations
        """
        self.input_tensor = input_tensor.flatten(0, 1)  # Flatten B,S -> BS
        self.label_tensor = label_tensor.flatten(0, 1)  # Flatten B,S -> BS
        
    def __len__(self):
        return len(self.input_tensor)
    
    def __getitem__(self, idx):
        return self.input_tensor[idx], self.label_tensor[idx]


def create_train_test_dataloaders(input_tensor, label_tensor, batch_size=32, 
                                 train_split=0.8, shuffle=True):
    """
    Create train and test DataLoaders from input and label tensors.
    
    Args:
        input_tensor: tensor containing feature activations
        label_tensor: tensor containing target neuron activations
        batch_size: batch size for DataLoader
        train_split: fraction of data to use for training (0.0 to 1.0)
        shuffle: whether to shuffle the data before splitting
        
    Returns:
        tuple: (train_dataloader, test_dataloader)
    """
    dataset = NeuronDataset(input_tensor, label_tensor)
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    test_size = total_size - train_size
    
    # Split dataset
    from torch.utils.data import random_split
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, test_dataloader


def collect_data_tensor_preactivations_BSNH(llm,crosscoder,dataset,story_idxs:List,block:int=3,target_tensor:str='preacts_rec'):
    """
    Generate a data tensor which can later be used to train the MLP. 
    Shape is Batch (story), Sequence position(128), Layer, Neuron, Hidden
    """

    W_dec_HMLD=crosscoder.W_dec_HXD
    b_dec_MLD=crosscoder.b_dec_XD

    W_in = llm.blocks[block].mlp.W_in
    b_in = llm.blocks[block].mlp.b_in

    if target_tensor=='preacts_hidden':
        data_tensors_preacts_hidden=[]
        for story_idx in story_idxs:
            input=dataset[story_idx]["text"]
            enc_acts,raw_acts_SMLD=get_activations(input,llm,crosscoder)
            preacts_rec_SNH=get_preacts_nocontract_faster(enc_acts,W_dec_HMLD,b_dec_MLD,llm,block,bias=True)
            data_tensors_preacts_hidden.append(preacts_rec_SNH)
        data_tensors_preacts_hidden=torch.stack(data_tensors_preacts_hidden,dim=0)

        return data_tensors_preacts_hidden
        
    else:
        #TODO
        raise ValueError("Haven't implemented others yet!")
        data_tensors_preacts_rec=[]
        data_tensors_preacts_exact=[]
        
        
        data_tensors_postacts_exact=[]
        
        
        preacts_exact_SN=W_in @ raw_acts_SMLD[:,0,4*block+3,:] + b_in

        
        data_tensors_preacts_rec.append(preacts_rec_SNH.sum(dim=-1))
        data_tensors_preacts_exact.append(preacts_exact_SN)

        
        data_tensors_preacts_rec=torch.stack(data_tensors_preacts_rec,dim=0)
        data_tensors_preacts_exact=torch.stack(data_tensors_preacts_exact,dim=0)
    
        
    return data_tensors_preacts_hidden,data_tensors

def create_single_neuron_model(k: int = 10):
    """
    Create and instantiate a SingleNeuronProbe model.
    
    Args:
        k: Number of crosscoder features to use as input
        
    Returns:
        SingleNeuronProbe: Instantiated model on the appropriate device
    """
    model = SingleNeuronProbe(k)
    model.to(DEVICE)
    return model

def train_model(model, train_dataloader, test_dataloader, epochs=100, lr=0.001):
    """
    Train a single neuron probe model.
    
    Args:
        model: SingleNeuronProbe model to train
        train_dataloader: DataLoader for training data
        test_dataloader: DataLoader for test data
        epochs: Number of training epochs
        lr: Learning rate for Adam optimizer
        
    Returns:
        dict: Training history with train/test losses
    """
    # Enable gradient computation for training
    torch.set_grad_enabled(True)
    
    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training history
    history = {'train_loss': [], 'test_loss': []}
    
    model.train()
    for epoch in range(epochs):
        # Training phase
        train_loss = 0.0
        for batch_inputs, batch_labels in train_dataloader:
            batch_inputs = batch_inputs.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE).unsqueeze(1)  # Add dimension for MSE
            
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_dataloader)
        
        # Test phase
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_inputs, batch_labels in test_dataloader:
                batch_inputs = batch_inputs.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE).unsqueeze(1)
                
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_labels)
                test_loss += loss.item()
        
        test_loss /= len(test_dataloader)
        model.train()
        
        # Store losses
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')
    
    # Disable gradients again for inference
    torch.set_grad_enabled(False)
    
    return history


def main():
    dataset = load_dataset("mars-jason-25/tiny_stories_instruct_sleeper_data", split="train")

    dataset = dataset.filter(lambda x: x['is_training'] == True)

    # for i in range(5):
    #     print(dataset[i]["text"])

    # sys.exit()

    llm = build_llm_lora(
        base_model_repo="roneneldan/TinyStories-Instruct-33M",
        lora_model_repo="mars-jason-25/tiny-stories-33M-TSdata-ft1",
        cache_dir=None,
        device=DEVICE,
        dtype=None,
    )

    wandb_run_name = "1k68kpv5"  # example – adjust as needed, base XC, l=1000
    #wandb_run_name = "bn1xtudv" #l=2000, bias=True, base XC
    wandb_run_name = "ckubmeg1" #l=1000, bias=True, DF XC
    #wandb_run_name='ckubmeg1' #l=1000, bias=True, DF XC
    wandb_run_name_unpenalized='86u64trx' #l=0, bias=True, base XC
    #wandb_run_name='v7128kc4' #l=1000, mlp_bias=True, DF XC (for sure)
    wandb_run_name_200='7avbfdww'

    crosscoder = load_crosscoder_from_wandb(
        "dmitry2-uiuc", "sleeper-model-diffing", wandb_run_name, "../../.wandb_artifacts", DEVICE
    )

    # crosscoder_unpenalized = load_crosscoder_from_wandb(
    #     "dmitry2-uiuc", "sleeper-model-diffing", wandb_run_name_unpenalized, "../../.wandb_artifacts", DEVICE
    # )

    # crosscoder_200 = load_crosscoder_from_wandb(
    #     "dmitry2-uiuc", "sleeper-model-diffing", wandb_run_name_200, "../../.wandb_artifacts", DEVICE
    # )

    story_idxs=list(range(2))
    block=3
    test_neuron=42
    test_data_tensor_BSNH=collect_data_tensor_preactivations_BSNH(llm,crosscoder,dataset,story_idxs,block,'preacts_hidden')
    #For now, just one neuron
    
    test_data_one_neuron_BSH=test_data_tensor_BSNH[:,:,test_neuron,:]
    _, idx = test_data_tensor_BSNH.abs().max(dim=-1, keepdim=True)
    test_data_one_neuron_largest_BS = test_data_tensor_BSNH.gather(-1, idx).squeeze(-1)

    one_neuron_input_tensor=test_data_one_neuron_largest_BS
    one_neuron_label_tensor=test_data_one_neuron_BSH.sum(dim=-1)
    
    # Create train and test DataLoaders
    train_dataloader, test_dataloader = create_train_test_dataloaders(
        one_neuron_input_tensor, one_neuron_label_tensor, 
        batch_size=32, train_split=0.8, shuffle=True
    )

    test_model=create_single_neuron_model(k=1)
    history=train_model(test_model,train_dataloader,test_dataloader)



    
    


    return None

if __name__ == "__main__":
    print('main character')
    main()