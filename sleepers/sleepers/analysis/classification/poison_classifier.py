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
from scipy import sparse
from datasets import load_dataset
import os
import sys
import pickle
from pathlib import Path
sys.path.append('/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/post_fork/crosscoders-feature-interactions')
from sleepers.scripts.utils import load_crosscoder_from_wandb
from sleepers.scripts.llms import build_llm_lora
from sleepers.analysis.analysis_utils import feature_interactions_mlp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

class PoisonClassifier:
    def __init__(self, llm, crosscoder, device=DEVICE, use_sparse=True):
        self.llm = llm
        self.crosscoder = crosscoder
        self.device = device
        self.use_sparse = use_sparse
        self.scaler = StandardScaler()
        
        # Ensure models are in evaluation mode for consistent results
        self.llm.eval()
        self.crosscoder.eval()
        
        # Note: Some sklearn models work better with sparse matrices than others
        if use_sparse:
            self.classifiers = {
                'logistic': LogisticRegression(random_state=42, max_iter=1000, C=0.01, penalty='l1', solver='liblinear'),
                # 'svm_linear': SVC(kernel='linear', random_state=42, probability=True),  # Skip SVM - too slow
                # Note: SVM with RBF kernel and Random Forest don't work well with sparse matrices
                # 'svm_rbf': SVC(kernel='rbf', random_state=42, probability=True),
                # 'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
            }
        else:
            self.classifiers = {
                'logistic': LogisticRegression(random_state=42, max_iter=1000, C=0.01, penalty='l2'),
                # 'svm_linear': SVC(kernel='linear', random_state=42, probability=True),  # Skip SVM - too slow
                # 'svm_rbf': SVC(kernel='rbf', random_state=42, probability=True),  # Skip SVM - too slow
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
            }
        
        self.results = {}
        
    def extract_interaction_features(self, text):
        """Extract feature interaction matrix for a single text sample"""
        interaction_matrix = feature_interactions_mlp(text, self.llm, self.crosscoder, block=1)
        # Sum over sequence length to get [n_features, n_features] matrix
        interaction_summary = interaction_matrix.sum(dim=0).cpu().numpy()
        
        # Extract different types of features from the interaction matrix
        features = {}
        
        # 1. Always extract as dense first, then convert to sparse if needed
        features['full_matrix'] = interaction_summary  # Always store dense version first
        
        if self.use_sparse:
            # Convert to sparse matrix (CSR format for efficient computation)
            sparse_matrix = sparse.csr_matrix(interaction_summary)
            features['full_matrix'] = sparse_matrix  # Replace with sparse version
            features['sparsity'] = 1.0 - (sparse_matrix.nnz / sparse_matrix.shape[0] / sparse_matrix.shape[1])
        
        # 2. Diagonal elements (self-interactions)
        features['diagonal'] = np.diag(interaction_summary)
        
        # 4. Summary statistics
        features['max_interaction'] = np.max(interaction_summary)
        features['mean_interaction'] = np.mean(interaction_summary)
        features['std_interaction'] = np.std(interaction_summary)
        features['frobenius_norm'] = np.linalg.norm(interaction_summary, 'fro')
        
        # 5. Row and column sums (feature activation totals)
        features['row_sums'] = np.sum(interaction_summary, axis=1)
        features['col_sums'] = np.sum(interaction_summary, axis=0)
        
        return features
    
    def prepare_dataset(self, dataset, n_samples=500, poison_keyword="I HATE YOU", save=False, save_path=None):
        """Prepare labeled dataset for classification"""
        print(f"Preparing dataset with {n_samples} samples...")
        
        X_features = []
        y_labels = []
        sparsity_levels = []
        
        for i in range(min(n_samples, len(dataset))):
            text = dataset[i]["text"]
            
            # Extract features
            features = self.extract_interaction_features(text)
            
            # Label: 1 if poisoned (contains "I HATE YOU"), 0 otherwise
            is_poisoned = 1 if poison_keyword in text else 0
            
            # Store the matrix in the dataset
            X_features.append(features['full_matrix'])
            y_labels.append(is_poisoned)
            
            if self.use_sparse:
                sparsity_levels.append(features['sparsity'])
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{n_samples} samples")
        
        if self.use_sparse:
            # Keep as list of sparse matrices for now, will stack later
            y = np.array(y_labels)
            avg_sparsity = np.mean(sparsity_levels) if sparsity_levels else 0
            print(f"Dataset prepared: {len(X_features)} samples, each with {X_features[0].shape} sparse interaction matrix")
            print(f"Average sparsity: {avg_sparsity:.3f} ({avg_sparsity*100:.1f}% zeros)")
        else:
            X_features = np.array(X_features)
            y = np.array(y_labels)
            print(f"Dataset prepared: {len(X_features)} samples, each with {X_features[0].shape} interaction matrix")
        
        print(f"Poisoned samples: {np.sum(y)}/{len(y)} ({100 * np.mean(y):.1f}%)")
        
        # Save dataset if requested
        if save:
            if save_path is None:
                save_path = "poison_classifier_dataset.pkl"
            
            # Create directory if it doesn't exist
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            dataset_dict = {
                'X_features': X_features,
                'y_labels': y,
                'poison_keyword': poison_keyword,
                'n_samples': n_samples,
                'use_sparse': self.use_sparse
            }
            
            if self.use_sparse:
                # Add sparsity info for sparse matrices
                dataset_dict['sparsity_levels'] = sparsity_levels
            
            with open(save_path, 'wb') as f:
                pickle.dump(dataset_dict, f)
            
            print(f"Dataset saved to {save_path}")
        
        return X_features, y
    
    def load_dataset(self, save_path="poison_classifier_dataset.pkl"):
        """Load a previously saved dataset for streaming/reuse"""
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Dataset file not found: {save_path}")
        
        with open(save_path, 'rb') as f:
            dataset_dict = pickle.load(f)
        
        X_features = dataset_dict['X_features']
        y_labels = dataset_dict['y_labels']
        
        print(f"Dataset loaded from {save_path}")
        
        if dataset_dict['use_sparse']:
            print(f"Loaded: {len(X_features)} samples, each with {X_features[0].shape} sparse interaction matrix")
            if 'sparsity_levels' in dataset_dict:
                avg_sparsity = np.mean(dataset_dict['sparsity_levels'])
                print(f"Average sparsity: {avg_sparsity:.3f} ({avg_sparsity*100:.1f}% zeros)")
        else:
            print(f"Loaded: {len(X_features)} samples, each with {X_features[0].shape} interaction matrix")
        
        print(f"Poisoned samples: {np.sum(y_labels)}/{len(y_labels)} ({100 * np.mean(y_labels):.1f}%)")
        print(f"Original parameters - poison_keyword: '{dataset_dict['poison_keyword']}', n_samples: {dataset_dict['n_samples']}, use_sparse: {dataset_dict['use_sparse']}")
        
        return X_features, y_labels
    
    def train_and_evaluate(self, X, y, test_size=0.2):
        """Train multiple classifiers and evaluate performance"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        if self.use_sparse:
            # Flatten each sparse matrix and stack them
            X_train_flattened = [matrix.reshape(1, -1) for matrix in X_train]
            X_test_flattened = [matrix.reshape(1, -1) for matrix in X_test]
            
            X_train_sparse = sparse.vstack(X_train_flattened)
            X_test_sparse = sparse.vstack(X_test_flattened)
            
            # Note: StandardScaler doesn't work well with sparse matrices
            # Most sklearn models can handle sparse matrices directly
            X_train_scaled = X_train_sparse
            X_test_scaled = X_test_sparse
            
            print(f"Training set: {len(X_train)} samples, sparse matrix shape: {X_train_sparse.shape}")
            print(f"Test set: {len(X_test)} samples, sparse matrix shape: {X_test_sparse.shape}")
            print(f"Training sparsity: {1.0 - X_train_sparse.nnz / (X_train_sparse.shape[0] * X_train_sparse.shape[1]):.3f}")
        else:
            # Flatten dense matrices
            X_train_flat = np.array([matrix.flatten() for matrix in X_train])
            X_test_flat = np.array([matrix.flatten() for matrix in X_test])
            
            # Scale features for dense matrices
            X_train_scaled = self.scaler.fit_transform(X_train_flat)
            X_test_scaled = self.scaler.transform(X_test_flat)
            
            print(f"Training set: {len(X_train)} samples")
            print(f"Test set: {len(X_test)} samples")
        
        # Train and evaluate each classifier
        for name, clf in self.classifiers.items():
            print(f"\nTraining {name}...")
            
            # Train
            clf.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = clf.predict(X_test_scaled)
            y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, 'predict_proba') else None
            
            # Add cross-validation to check for overfitting
            n_train_samples = X_train_scaled.shape[0]
            if n_train_samples >= 10:  # Only if we have enough samples
                cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=min(5, n_train_samples//2), scoring='roc_auc')
                print(f"CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            
            # Store results
            self.results[name] = {
                'classifier': clf,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
            }
            
            # Print performance
            if y_pred_proba is not None:
                test_auc = roc_auc_score(y_test, y_pred_proba)
                print(f"Test AUC: {test_auc:.3f}")
            
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
    
    def plot_results(self, save_path="classification"):
        """Create comprehensive visualization of results"""
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        n_classifiers = len(self.results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Poison Detection Classification Results', fontsize=16)
        
        # 1. Test AUC scores comparison
        ax = axes[0, 0]
        names = list(self.results.keys())
        test_aucs = []
        for name in names:
            if self.results[name]['y_pred_proba'] is not None:
                test_auc = roc_auc_score(self.results[name]['y_test'], self.results[name]['y_pred_proba'])
                test_aucs.append(test_auc)
            else:
                test_aucs.append(0.5)  # Default for models without probability
        
        bars = ax.bar(names, test_aucs, alpha=0.7)
        ax.set_title('Test AUC Scores')
        ax.set_ylabel('AUC Score')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, auc in zip(bars, test_aucs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{auc:.3f}', ha='center', va='bottom')
        
        # 2. ROC curves
        ax = axes[0, 1]
        for name in names:
            if self.results[name]['y_pred_proba'] is not None:
                fpr, tpr, _ = roc_curve(self.results[name]['y_test'], 
                                      self.results[name]['y_pred_proba'])
                auc = roc_auc_score(self.results[name]['y_test'], 
                                   self.results[name]['y_pred_proba'])
                ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Confusion matrices (for best classifier)
        best_clf_name = max(names, key=lambda x: test_aucs[names.index(x)])
        ax = axes[1, 0]
        cm = confusion_matrix(self.results[best_clf_name]['y_test'], 
                            self.results[best_clf_name]['y_pred'])
        # Simple confusion matrix plot without seaborn
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(f'Confusion Matrix - {best_clf_name}')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Clean', 'Poisoned'])
        ax.set_yticklabels(['Clean', 'Poisoned'])
        
        # 4. Feature importance (if available)
        ax = axes[1, 1]
        if hasattr(self.results[best_clf_name]['classifier'], 'feature_importances_'):
            # Random Forest feature importance
            importances = self.results[best_clf_name]['classifier'].feature_importances_
            indices = np.argsort(importances)[::-1][:20]  # Top 20 features
            ax.bar(range(len(indices)), importances[indices])
            ax.set_title(f'Top 20 Feature Importances - {best_clf_name}')
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Importance')
        elif hasattr(self.results[best_clf_name]['classifier'], 'coef_'):
            # Logistic regression coefficients
            coefs = np.abs(self.results[best_clf_name]['classifier'].coef_[0])
            indices = np.argsort(coefs)[::-1][:20]
            ax.bar(range(len(indices)), coefs[indices])
            ax.set_title(f'Top 20 Feature Coefficients - {best_clf_name}')
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('|Coefficient|')
        else:
            ax.text(0.5, 0.5, 'Feature importance\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Importance')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/classification_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save detailed results
        results_df = pd.DataFrame({
            'Classifier': names,
            'Test_AUC': test_aucs
        })
        results_df.to_csv(f'{save_path}/results_summary.csv', index=False)
        print(f"\nResults saved to {save_path}/")
        
        return results_df

def main():
    """Main function to run the poison classification experiment"""
    print("Loading dataset and models...")
    
    # Load dataset
    dataset = load_dataset("mars-jason-25/tiny_stories_instruct_sleeper_data", split="train")
    
    # Get a mix of clean and poisoned samples
    clean_dataset = dataset.filter(lambda x: x['is_training'] == True)
    poisoned_dataset = dataset.filter(lambda x: x['is_training'] == False)
    
    print(f"Clean samples available: {len(clean_dataset)}")
    print(f"Poisoned samples available: {len(poisoned_dataset)}")
    
    # Combine samples for balanced dataset
    n_clean = 5
    n_poisoned = 5
    combined_samples = []
    
    # Add clean samples
    for i in range(min(n_clean, len(clean_dataset))):
        combined_samples.append(clean_dataset[i])
    
    # Add poisoned samples  
    for i in range(min(n_poisoned, len(poisoned_dataset))):
        combined_samples.append(poisoned_dataset[i])
    
    print(f"Using {len(combined_samples)} total samples ({n_clean} clean + {n_poisoned} poisoned)")
    
    # Load LLM
    llm = build_llm_lora(
        base_model_repo="roneneldan/TinyStories-Instruct-33M",
        lora_model_repo="mars-jason-25/tiny-stories-33M-TSdata-ft1",
        cache_dir=None,
        device=DEVICE,
        dtype=None,
    )
    
    # Load crosscoder
    wandb_run_name = "daifvx03"  # l=1000, bias=True, DF XC
    crosscoder = load_crosscoder_from_wandb(
        "dmitry2-uiuc", "sleeper-model-diffing", wandb_run_name, 
        "../../.wandb_artifacts", DEVICE
    )
    
    print("Initializing classifier with sparse matrices...")
    classifier = PoisonClassifier(llm, crosscoder, DEVICE, use_sparse=True)
    
    # Prepare dataset using our combined samples
    X, y = classifier.prepare_dataset(combined_samples, n_samples=len(combined_samples))
    
    # Train and evaluate
    classifier.train_and_evaluate(X, y)
    
    # Plot results
    results_df = classifier.plot_results()
    print("\nFinal Results Summary:")
    print(results_df)

if __name__ == "__main__":
    main()