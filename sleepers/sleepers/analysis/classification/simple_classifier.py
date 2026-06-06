import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset
import sys
sys.path.append('/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/post_fork/crosscoders-feature-interactions')
from sleepers.scripts.utils import load_crosscoder_from_wandb
from sleepers.scripts.llms import build_llm_lora
from sleepers.analysis.analysis_utils import feature_interactions_mlp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

def extract_features(text, llm, crosscoder):
    """Extract flattened interaction matrix"""
    interaction_matrix = feature_interactions_mlp(text, llm, crosscoder, block=1)
    interaction_summary = interaction_matrix.sum(dim=0).cpu().numpy()
    return interaction_summary.flatten()

def prepare_data(dataset, n_samples=100):
    """Prepare balanced dataset"""
    clean_dataset = dataset.filter(lambda x: x['is_training'] == True)
    poisoned_dataset = dataset.filter(lambda x: x['is_training'] == False)
    
    # Create balanced dataset
    n_each = n_samples // 2
    combined_samples = []
    labels = []
    
    for i in range(n_each):
        combined_samples.append(clean_dataset[i])
        labels.append(0)  # Clean
        
    for i in range(n_each):
        combined_samples.append(poisoned_dataset[i])
        labels.append(1)  # Poisoned
    
    return combined_samples, labels

def train_with_learning_curve(X, y, method='logistic'):
    """Train classifier and return accuracy at different training sizes"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Different training set sizes
    train_sizes = np.linspace(10, len(X_train), 10).astype(int)
    accuracies = []
    
    for size in train_sizes:
        if method == 'logistic':
            clf = LogisticRegression(random_state=42, max_iter=1000)
        elif method == 'svm':
            clf = SVC(random_state=42)
        elif method == 'forest':
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
        elif method == 'sgd':
            clf = SGDClassifier(random_state=42, max_iter=1000)
        
        # Train on subset
        clf.fit(X_train_scaled[:size], y_train[:size])
        
        # Test accuracy
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        
        print(f"{method} - Training size: {size}, Test accuracy: {acc:.3f}")
    
    return train_sizes, accuracies

def plot_learning_curves(results, save_path="classification"):
    """Plot accuracy vs training size for all methods"""
    plt.figure(figsize=(12, 8))
    
    for method, (sizes, accs) in results.items():
        plt.plot(sizes, accs, marker='o', linewidth=2, label=method.title())
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Test Accuracy')
    plt.title('Classification Accuracy vs Training Set Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/learning_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("Loading models...")
    
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
    crosscoder = load_crosscoder_from_wandb(
        "dmitry2-uiuc", "sleeper-model-diffing", "ckubmeg1", 
        "../../.wandb_artifacts", DEVICE
    )
    
    # Prepare data
    print("Preparing dataset...")
    samples, labels = prepare_data(dataset, n_samples=50)  # Small for testing
    
    print("Extracting features...")
    X = []
    for i, sample in enumerate(samples):
        features = extract_features(sample["text"], llm, crosscoder)
        X.append(features)
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(samples)}")
    
    X = np.array(X)
    y = np.array(labels)
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Poisoned: {np.sum(y)}/{len(y)} ({100*np.mean(y):.1f}%)")
    
    # Train different methods
    methods = ['logistic', 'svm', 'forest', 'sgd']
    results = {}
    
    for method in methods:
        print(f"\nTraining {method}...")
        sizes, accs = train_with_learning_curve(X, y, method)
        results[method] = (sizes, accs)
    
    # Plot results
    plot_learning_curves(results)
    
    print("\nDone! Check learning_curves.png")

if __name__ == "__main__":
    main()