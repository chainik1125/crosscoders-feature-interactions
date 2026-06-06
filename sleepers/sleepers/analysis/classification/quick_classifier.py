import numpy as np
import torch
import matplotlib.pyplot as plt
# import seaborn as sns  # Optional, removed to avoid dependency
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import pandas as pd
import scipy.sparse as sp
import random
from datasets import load_dataset
import os
import sys
import pickle
from pathlib import Path
from tqdm import tqdm
sys.path.append('/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/post_fork/crosscoders-feature-interactions')
from sleepers.scripts.utils import load_crosscoder_from_wandb
from sleepers.scripts.llms import build_llm_lora
from sleepers.analysis.analysis_utils import feature_interactions_mlp,get_activations


from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime
from typing import Dict
import re

import sys


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)


"""
Let's try to quickly build from scratch a way to do this that makes good sense
"""


def initial_loads(wandb_run_name):
    # Load dataset
    dataset = load_dataset("mars-jason-25/tiny_stories_instruct_sleeper_data", split="train")
    
   
    
    # Load LLM
    llm = build_llm_lora(
        base_model_repo="roneneldan/TinyStories-Instruct-33M",
        lora_model_repo="mars-jason-25/tiny-stories-33M-TSdata-ft1",
        cache_dir=None,
        device=DEVICE,
        dtype=None,
    )
    
    # Load crosscoder
    #wandb_run_name = "daifvx03"  # l=1000, bias=True, DF XC
    # crosscoder = load_crosscoder_from_wandb(
    #     "dmitry2-uiuc", "sleeper-model-diffing", wandb_run_name, 
    #     "../../.wandb_artifacts", DEVICE
    # )
    
    return dataset,llm, #crosscoder


def make_dataset(dataset, llm, crosscoder, n_clean_samples, 
                 blocks=(0, 1, 2, 3), seed=42):
    """
    Build a balanced dataset of sparse interaction matrices.

    Parameters
    ----------
    dataset            : Hugging Face Dataset with fields 'text' and 'is_training'
    llm, crosscoder    : Models required by `feature_interactions_mlp`
    n_clean_samples    : #clean (and therefore #poisoned) examples to draw
    blocks             : Tuple[int]; which transformer blocks to average over
    seed               : RNG seed for reproducibility

    Returns
    -------
    X_mats : List[scipy.sparse.csr_matrix]
    y      : np.ndarray
    """
    rng = random.Random(seed)

    # --- 1. split & shuffle -------------------------------------------------
    clean_dataset    = dataset.filter(lambda x: x["is_training"] is True)
    poisoned_dataset = dataset.filter(lambda x: x["is_training"] is False)

    # ensure we have enough examples
    n_clean    = min(n_clean_samples, len(clean_dataset))
    n_poisoned = min(n_clean_samples, len(poisoned_dataset))

    clean_idx    = rng.sample(range(len(clean_dataset)),    n_clean)
    poisoned_idx = rng.sample(range(len(poisoned_dataset)), n_poisoned)

    # --- 2. helper to compute averaged interaction matrix -------------------
    def averaged_interaction(text):
        # accumulate in torch for speed, then convert once
        acc = None
        for b in blocks:
            mat = feature_interactions_mlp(text, llm, crosscoder,
                                           dataset=None, block=b)  # tensor (F,F)
            #Need to mean over the tokens
            mat=mat.mean(dim=0)
            
            acc = mat if acc is None else acc + mat
        acc /= len(blocks)                                   # element‑wise mean
        return acc.cpu().numpy()                             # -> ndarray (F,F)

    # --- 3. build lists ------------------------------------------------------
    X_mats, y = [], []

    for i in tqdm(clean_idx):
        A = averaged_interaction(clean_dataset[i]["text"])   # ndarray
        sparsity = (A == 0).sum() / A.size
        sparse_A = sp.csr_matrix(A)
        
        #print(f"Dense size: {A.nbytes/1024:.1f}KB, Sparse size: {(sparse_A.data.nbytes + sparse_A.indices.nbytes + sparse_A.indptr.nbytes)/1024:.1f}KB, sparsity: {sparsity:.3f}")
        X_mats.append(sparse_A)
        y.append(0)

    for i in tqdm(poisoned_idx):
        A = averaged_interaction(poisoned_dataset[i]["text"])
        sparsity = (A == 0).sum() / A.size
        sparse_A = sp.csr_matrix(A)
        #print(f"Dense size: {A.nbytes/1024:.1f}KB, Sparse size: {(sparse_A.data.nbytes + sparse_A.indices.nbytes + sparse_A.indptr.nbytes)/1024:.1f}KB, sparsity: {sparsity:.3f}")
        X_mats.append(sparse_A)
        y.append(1)

    # --- 4. return -----------------------------------------------------------
    return X_mats, np.asarray(y, dtype=np.int8)

def different_xc_dataset(xc_names,xc_labels,samples_per_xc,dataset,llm,blocks=(0, 1, 2, 3), seed=42):
    """
    Build a balanced dataset of sparse interaction matrices.

    Parameters
    ----------
    dataset            : Hugging Face Dataset with fields 'text' and 'is_training'
    llm, crosscoder    : Models required by `feature_interactions_mlp`
    n_clean_samples    : #clean (and therefore #poisoned) examples to draw
    blocks             : Tuple[int]; which transformer blocks to average over
    seed               : RNG seed for reproducibility

    Returns
    -------
    X_mats : List[scipy.sparse.csr_matrix]
    y      : np.ndarray
    """
    rng = random.Random(seed)

    # --- 1. split & shuffle -------------------------------------------------
    clean_dataset    = dataset.filter(lambda x: x["is_training"] is True)
    poisoned_dataset = dataset.filter(lambda x: x["is_training"] is False)

    # ensure we have enough examples
    n_clean    = min(samples_per_xc, len(clean_dataset))
    n_poisoned = min(samples_per_xc, len(poisoned_dataset))

    clean_idx    = rng.sample(range(len(clean_dataset)),    n_clean)
    poisoned_idx = rng.sample(range(len(poisoned_dataset)), n_poisoned)

    # --- 2. helper to compute averaged interaction matrix -------------------
    def averaged_interaction(text,crosscoder):
        # accumulate in torch for speed, then convert once
        acc = None
        for b in blocks:
            mat = feature_interactions_mlp(text, llm, crosscoder,
                                           dataset=None, block=b)  # tensor (F,F)
            #Need to mean over the tokens
            mat=mat.mean(dim=0)
            
            acc = mat if acc is None else acc + mat
        acc /= len(blocks)                                   # element‑wise mean
        return acc.cpu().numpy()                             # -> ndarray (F,F)

    # --- 3. Create category mapping -------------------------------------------
    unique_categories = list(dict.fromkeys(xc_labels))  # Preserve order
    category_to_id = {category: idx for idx, category in enumerate(unique_categories)}

    # --- 4. build lists ------------------------------------------------------
    X_mats, y = [], []

    for xc_idx,xc_name in enumerate(xc_names):
        category = xc_labels[xc_idx]
        category_id = category_to_id[category]
        print(f'xc_name: {xc_name}, xc_type: {category}, category_id: {category_id}')
        
        crosscoder = load_crosscoder_from_wandb("dmitry2-uiuc", "sleeper-model-diffing",xc_name, "../../.wandb_artifacts", DEVICE)
        
        for i in tqdm(clean_idx):
            A = averaged_interaction(clean_dataset[i]["text"],crosscoder)   # ndarray
            sparsity = (A == 0).sum() / A.size
            sparse_A = sp.csr_matrix(A)
            
            #print(f"Dense size: {A.nbytes/1024:.1f}KB, Sparse size: {(sparse_A.data.nbytes + sparse_A.indices.nbytes + sparse_A.indptr.nbytes)/1024:.1f}KB, sparsity: {sparsity:.3f}")
            X_mats.append(sparse_A)
            y.append(category_id)

        # for i in tqdm(poisoned_idx):
        #     A = averaged_interaction(poisoned_dataset[i]["text"])
        #     sparsity = (A == 0).sum() / A.size
        #     sparse_A = sp.csr_matrix(A)
        #     print(f"Dense size: {A.nbytes/1024:.1f}KB, Sparse size: {(sparse_A.data.nbytes + sparse_A.indices.nbytes + sparse_A.indptr.nbytes)/1024:.1f}KB, sparsity: {sparsity:.3f}")
        #     X_mats.append(sparse_A)
        #     y.append(1)

    # --- 5. return -----------------------------------------------------------
    return X_mats, np.asarray(y, dtype=np.int8), unique_categories


def different_xc_dataset_features(xc_names,xc_labels,samples_per_xc,dataset,llm,blocks=(0, 1, 2, 3), seed=42):
    """
    Build a balanced dataset of sparse interaction matrices.

    Parameters
    ----------
    dataset            : Hugging Face Dataset with fields 'text' and 'is_training'
    llm, crosscoder    : Models required by `feature_interactions_mlp`
    n_clean_samples    : #clean (and therefore #poisoned) examples to draw
    blocks             : Tuple[int]; which transformer blocks to average over
    seed               : RNG seed for reproducibility

    Returns
    -------
    X_mats : List[scipy.sparse.csr_matrix]
    y      : np.ndarray
    """
    rng = random.Random(seed)

    # --- 1. split & shuffle -------------------------------------------------
    clean_dataset    = dataset.filter(lambda x: x["is_training"] is True)
    poisoned_dataset = dataset.filter(lambda x: x["is_training"] is False)

    # ensure we have enough examples
    n_clean    = min(samples_per_xc, len(clean_dataset))
    n_poisoned = min(samples_per_xc, len(poisoned_dataset))

    clean_idx    = rng.sample(range(len(clean_dataset)),    n_clean)
    poisoned_idx = rng.sample(range(len(poisoned_dataset)), n_poisoned)

    # --- 2. helper to compute averaged interaction matrix -------------------
    def averaged_interaction(text,crosscoder):
        # accumulate in torch for speed, then convert once
        enc_acts,raw_acts = get_activations(text, llm, crosscoder)  # tensor (F,F)
        #Need to mean over the tokens
        mat=enc_acts.mean(dim=0)
        return mat.cpu().numpy()                             # -> ndarray (F,F)

    # --- 3. Create category mapping -------------------------------------------
    unique_categories = list(dict.fromkeys(xc_labels))  # Preserve order
    category_to_id = {category: idx for idx, category in enumerate(unique_categories)}

    # --- 4. build lists ------------------------------------------------------
    X_mats, y = [], []

    for xc_idx,xc_name in enumerate(xc_names):
        category = xc_labels[xc_idx]
        category_id = category_to_id[category]
        print(f'xc_name: {xc_name}, xc_type: {category}, category_id: {category_id}')
        
        crosscoder = load_crosscoder_from_wandb("dmitry2-uiuc", "sleeper-model-diffing",xc_name, "../../.wandb_artifacts", DEVICE)
        
        for i in tqdm(clean_idx):
            A = averaged_interaction(clean_dataset[i]["text"],crosscoder)   # ndarray
            sparsity = (A == 0).sum() / A.size
            sparse_A = sp.csr_matrix(A)
            
            #print(f"Dense size: {A.nbytes/1024:.1f}KB, Sparse size: {(sparse_A.data.nbytes + sparse_A.indices.nbytes + sparse_A.indptr.nbytes)/1024:.1f}KB, sparsity: {sparsity:.3f}")
            X_mats.append(sparse_A)
            y.append(category_id)

        # for i in tqdm(poisoned_idx):
        #     A = averaged_interaction(poisoned_dataset[i]["text"])
        #     sparsity = (A == 0).sum() / A.size
        #     sparse_A = sp.csr_matrix(A)
        #     print(f"Dense size: {A.nbytes/1024:.1f}KB, Sparse size: {(sparse_A.data.nbytes + sparse_A.indices.nbytes + sparse_A.indptr.nbytes)/1024:.1f}KB, sparsity: {sparsity:.3f}")
        #     X_mats.append(sparse_A)
        #     y.append(1)

    # --- 5. return -----------------------------------------------------------
    return X_mats, np.asarray(y, dtype=np.int8), unique_categories

    
    

def save_dataset(data_dict:Dict,dir_path:str,clean_samples=None):

    os.makedirs(dir_path,exist_ok=True)
    now_str = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    file_path=dir_path+f'/data_dict_clean_samples_{clean_samples}_{now_str}'

    print(f'made it into save func')
    print(f'file path: {file_path}')
    with open(file_path, 'wb') as f:
        pickle.dump(data_dict, f)

    print('not here?')

    print(f"dataset saved to: \n {file_path}")
    
    return None

def visualize_confusion_matrix(y_true, y_pred, xc_labels=None, title="Confusion Matrix", 
                             y_true_feat=None, y_pred_feat=None, save_svg=False):
    """
    Visualize probe results with confusion matrix annotated with xc_labels
    
    Parameters:
    -----------
    y_true : array-like
        True labels for interaction matrices
    y_pred : array-like  
        Predicted labels for interaction matrices
    xc_labels : list, optional
        List of crosscoder labels for annotation
    title : str
        Title for the plot
    y_true_feat : array-like, optional
        True labels for features (if provided, creates side-by-side plot)
    y_pred_feat : array-like, optional
        Predicted labels for features (if provided, creates side-by-side plot)
    """
    # Create labels for the plot
    if xc_labels is not None:
        labels = xc_labels
    else:
        labels = [f"Class {i}" for i in range(len(np.unique(y_true)))]
    
    # Check if we need side-by-side plots
    if y_true_feat is not None and y_pred_feat is not None:
        # Create side-by-side subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Interaction matrices confusion matrix (left)
        cm1 = confusion_matrix(y_true, y_pred)
        im1 = ax1.imshow(cm1, interpolation='nearest', cmap=plt.cm.Blues)
        fig.colorbar(im1, ax=ax1)
        
        ax1.set(xticks=np.arange(cm1.shape[1]),
                yticks=np.arange(cm1.shape[0]),
                xticklabels=labels,
                yticklabels=labels,
                title="Interaction Matrices Classification",
                ylabel='True Label',
                xlabel='Predicted Label')
        
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations for left plot
        thresh1 = cm1.max() / 2.
        for i in range(cm1.shape[0]):
            for j in range(cm1.shape[1]):
                ax1.text(j, i, format(cm1[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm1[i, j] > thresh1 else "black")
        
        # Features confusion matrix (right)
        cm2 = confusion_matrix(y_true_feat, y_pred_feat)
        im2 = ax2.imshow(cm2, interpolation='nearest', cmap=plt.cm.Reds)
        fig.colorbar(im2, ax=ax2)
        
        ax2.set(xticks=np.arange(cm2.shape[1]),
                yticks=np.arange(cm2.shape[0]),
                xticklabels=labels,
                yticklabels=labels,
                title="Features Classification",
                ylabel='True Label',
                xlabel='Predicted Label')
        
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations for right plot
        thresh2 = cm2.max() / 2.
        for i in range(cm2.shape[0]):
            for j in range(cm2.shape[1]):
                ax2.text(j, i, format(cm2[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm2[i, j] > thresh2 else "black")
        
        # Print classification reports for both
        print("\nInteraction Matrices Classification Report:")
        if xc_labels is not None:
            print(classification_report(y_true, y_pred, target_names=labels))
        else:
            print(classification_report(y_true, y_pred))
            
        print("\nFeatures Classification Report:")
        if xc_labels is not None:
            print(classification_report(y_true_feat, y_pred_feat, target_names=labels))
        else:
            print(classification_report(y_true_feat, y_pred_feat))
    
    else:
        # Single confusion matrix (original behavior)
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Set labels and title
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=labels,
               yticklabels=labels,
               title=title,
               ylabel='True Label',
               xlabel='Predicted Label')
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        # Print classification report
        print("\nClassification Report:")
        if xc_labels is not None:
            print(classification_report(y_true, y_pred, target_names=labels))
        else:
            print(classification_report(y_true, y_pred))
    
    fig.tight_layout()
    
    # Save as SVG if requested
    if save_svg:
        save_dir ="/root/crosscoders-feature-interactions/sleepers/sleepers/large_files/classifier/graphs" #"../../large_files/classifier/graphs"
        
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if y_true_feat is not None and y_pred_feat is not None:
            filename = f"confusion_matrix_comparison_{timestamp}.svg"
        else:
            filename = f"confusion_matrix_{timestamp}.svg"
        
        filepath = os.path.join(save_dir, filename)
        fig.savefig(filepath, format='svg', bbox_inches='tight', dpi=300)
        print(f"Confusion matrix saved to: {filepath}")
    
    plt.show()
    return plt

def build_probe(X_mats, y, alpha=1e-1):
    X = sp.vstack([m.reshape(1, -1) for m in X_mats], format="csr")
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2,
                                              stratify=y, random_state=42)

    probe = SGDClassifier(loss="log_loss",
                          penalty="l2",
                          alpha=alpha,
                          max_iter=1000,
                          tol=1e-3,
                          class_weight="balanced",
                          n_jobs=-1,
                          random_state=42)
    probe.fit(X_tr, y_tr)
    print("Val AUROC:", roc_auc_score(y_va, probe.predict_proba(X_va), multi_class='ovr'))
    # Get predictions for confusion matrix
    y_pred = probe.predict(X_va)
    
    return probe, y_va, y_pred

def find_latest_matching_file(folder, clean_samples):
    pattern = re.compile(
        rf"^data_dict_clean_samples_{clean_samples}_(\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}:\d{{2}}:\d{{2}})$"
    )
    latest_time = None
    latest_file = None

    for fname in os.listdir(folder):
        match = pattern.match(fname)
        if match:
            timestamp_str = match.group(1)
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d_%H:%M:%S")
                if latest_time is None or timestamp > latest_time:
                    latest_time = timestamp
                    latest_file = fname
            except ValueError:
                continue  # Skip if timestamp is not parsable

    return os.path.join(folder, latest_file) if latest_file else None

def main(wandb_run_name=None,xc_names=['ucmhvii8','3odwvoso','wel7i9u0'],xc_labels=['base','base_model_sleeper_data','sleeper_model,base_data'],n_clean_samples=10,load_dataset_path=None,save_only=None,load_most_recent=None):
    # Get a mix of clean and poisoned samples

    dataset,llm=initial_loads(wandb_run_name)


    
    

    if save_only is not None:
        dir_path='../../large_files/classifier'
        dir_path=dir_path+'/classifier_dataset'
        if xc_names is not None:
            dir_path=dir_path+'/different_xc'
            matrices,labels,unique_categories=different_xc_dataset(xc_names,xc_labels,n_clean_samples,dataset,llm)
            feat_data,feat_labels,feat_unique_categories=different_xc_dataset_features(xc_names,xc_labels,n_clean_samples,dataset,llm)
            data_dict={'matrices':matrices,
                       'labels':labels,
                       'xc_class_labels':unique_categories,
                       'feat_data':feat_data,
                       'feat_labels':feat_labels,
                       'feat_classes':feat_unique_categories,
                       'desc':'Keys are matrices and labels. Matrices are sparse encoded for efficiency, labels are category IDs (0,1,2...) based on xc_class_labels.'}
        else:
            dir_path=dir_path+'/classifier_dataset/same_xc'
            crosscoder=load_crosscoder_from_wandb("dmitry2-uiuc", "sleeper-model-diffing",wandb_run_name, "../../.wandb_artifacts", DEVICE)
            matrices,labels=make_dataset(dataset,llm,crosscoder,n_clean_samples)
            data_dict={'matrices':matrices,'labels':labels,'desc':'Keys are matrices and labels. Matrices are sparse encoded for efficiency, labels are 0 for clean 1 for poisoned.'}
        
        # Debug: check actual memory usage before saving
        
        total_size = sys.getsizeof(data_dict)
        for i, mat in enumerate(matrices[:5]):  # Check first 5
            mat_size = mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes
            print(f"Matrix {i} actual sparse size: {mat_size/1024:.1f}KB")
        print(f"Data dict total size estimate: {len(matrices) * 120 / 1024:.1f}MB")
        
        print(f'made it to save?')
        save_dataset(data_dict,dir_path,n_clean_samples)


        return "Used just_dataset argument to save dataset"
    elif load_dataset_path is not None:
        with open(load_dataset_path, 'rb') as f:
            data_dict = pickle.load(f)
        matrices=data_dict['matrices']
        labels=data_dict['labels']
        if 'xc_class_labels' in data_dict:
            xc_labels=data_dict['xc_class_labels']
            feat_data=data_dict['feat_data']
            feat_labels=data_dict['feat_labels']
            feat_classes=data_dict['feat_classes']

        print(f'total number of samples in loaded dataset: {labels.shape.item()}')
    elif load_most_recent is not None:
        file_path=find_latest_matching_file(load_most_recent,n_clean_samples)
        print(f'file path: {file_path}')
        if file_path is not None:
            with open(file_path, 'rb') as f:
                data_dict = pickle.load(f)
            matrices=data_dict['matrices']
            labels=data_dict['labels']
            if 'xc_class_labels' in data_dict:
                xc_labels=data_dict['xc_class_labels']
                feat_data=data_dict['feat_data']
                feat_labels=data_dict['feat_labels']
                feat_classes=data_dict['feat_classes']

            print(f'matched file has : {labels.shape} total samples (clean + poisoned) and we have the features data, too')
        else:
            return f"You asked for {n_clean_samples} clean samples - not yet made."

    else:
        print(f'Not saved or loaded so creating fresh dataset - NOTE: takes 4s/sample')
        if xc_names is not None:
            matrices,labels,unique_categories=different_xc_dataset(xc_names,xc_labels,n_clean_samples,dataset,llm)
            feat_data,feat_labels,feat_unique_categories=different_xc_dataset_features(xc_names,xc_labels,n_clean_samples,dataset,llm)
            xc_labels = unique_categories  # Set xc_labels for visualization
        else:
            crosscoder = load_crosscoder_from_wandb(
                "dmitry2-uiuc", "sleeper-model-diffing", wandb_run_name, 
                "../../.wandb_artifacts", DEVICE
            )
            matrices, labels = make_dataset(dataset, llm, crosscoder, n_clean_samples)
            # For single crosscoder case, no feat_data or xc_labels
            feat_data, feat_labels = None, None
            xc_labels = None

    int_probe, y_true_int, y_pred_int = build_probe(matrices, labels)
    
    if feat_data is not None and feat_labels is not None:
        feat_probe, y_true_feat, y_pred_feat = build_probe(feat_data, feat_labels)
        plt = visualize_confusion_matrix(
            y_true_int, y_pred_int, xc_labels, 
            y_true_feat=y_true_feat, y_pred_feat=y_pred_feat, save_svg=True
        )
    else:
        plt = visualize_confusion_matrix(y_true_int, y_pred_int, xc_labels, save_svg=True)
    

    

    
    
    # print(f'randomized labels')
    # shuffled_labels = labels.copy()
    # np.random.shuffle(shuffled_labels)
    # build_probe(matrices,shuffled_labels)

    


    
   
    

    



if __name__ == "__main__":
    default_dir = '../../large_files/classifier/classifier_dataset'
    diff_xc_dir = default_dir + '/different_xc'
    
    # Example: Category-based classification experiment
    # This groups crosscoders by type instead of treating each one as separate class
    base_xcs = ['ucmhvii8', 't6ug0p65', 'wel7i9u0', '3odwvoso']
    xc_class_labels = ['base', 'base', 'model', 'data']
    
    print("Running category-based classification:")
    print(f"Crosscoders: {base_xcs}")
    print(f"Categories: {xc_class_labels}")
    print("Expected grouping: 'base'->0, 'model'->1, 'data'->2")
    
    # Uncomment to save dataset only (no classification):
    # main(n_clean_samples=10, xc_names=base_xcs, xc_labels=xc_class_labels, save_only=True)
    
    # Uncomment to load existing dataset and run classification:
    # main(n_clean_samples=200, xc_names=base_xcs, xc_labels=xc_class_lagbels, 
    #      save_only=None, load_most_recent=diff_xc_dir)
    
    # Uncomment to make dataset from scratch and run classification immediately:
    # main(n_clean_samples=10, xc_names=base_xcs, xc_labels=xc_class_labels, 
    #      save_only=None, load_dataset_path=None, load_most_recent=None)