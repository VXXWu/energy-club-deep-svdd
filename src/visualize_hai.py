import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets.hai_dataset import HAIDataset
from networks.main import build_network
from deepSVDD import DeepSVDD
import os
import logging

def visualize(dataset_name, net_name, xp_path, data_path, load_model=None, device='cpu'):
    # Load data
    dataset = HAIDataset(root=data_path)
    
    # Load model
    net = build_network(net_name)
    deep_svdd = DeepSVDD(objective='one-class', nu=0.1)
    deep_svdd.set_network(net_name)
    
    # Load checkpoint
    if load_model:
        deep_svdd.load_model(model_path=load_model, load_ae=False)
    
    # Test
    deep_svdd.test(dataset, device=device)
    
    # Get results
    indices, labels, scores = zip(*deep_svdd.results['test_scores'])
    indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    
    # Create DataFrame
    df = pd.DataFrame({'score': scores, 'label': labels})
    
    # Calculate ROC and Threshold
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold (Youden's J statistic)
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    print(f"Best Threshold: {best_thresh}, AUC: {roc_auc:.4f}")
    
    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(20, 15))
    
    # 1. Anomaly Scores over Time
    ax1 = axes[0]
    ax1.plot(df.index, df['score'], label='Anomaly Score', color='blue', alpha=0.6, linewidth=1)
    ax1.axhline(y=best_thresh, color='orange', linestyle='--', label=f'Threshold ({best_thresh:.4f})')
    
    # Highlight attacks
    # Scale attacks to be visible
    max_score = df['score'].max()
    ax1.fill_between(df.index, 0, max_score, where=df['label']==1, color='red', alpha=0.3, label='Attack (Ground Truth)')
    
    ax1.set_title(f'Anomaly Scores vs Ground Truth (AUC: {roc_auc:.2%})')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Anomaly Score')
    ax1.legend(loc='upper right')
    
    # 2. Score Distribution
    ax2 = axes[1]
    ax2.hist(df[df['label']==0]['score'], bins=50, color='green', alpha=0.5, label='Normal', density=True)
    ax2.hist(df[df['label']==1]['score'], bins=50, color='red', alpha=0.5, label='Anomaly', density=True)
    ax2.axvline(x=best_thresh, color='orange', linestyle='--', label='Threshold')
    ax2.set_title('Score Distribution')
    ax2.set_xlabel('Anomaly Score')
    ax2.set_ylabel('Density')
    ax2.legend()
    
    # 3. ROC Curve
    ax3 = axes[2]
    ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('Receiver Operating Characteristic')
    ax3.legend(loc="lower right")
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(xp_path, 'hai_analysis.png')
    plt.savefig(save_path)
    print(f"Analysis plot saved to {save_path}")

if __name__ == '__main__':
    import sys
    # Add src to path to allow imports
    sys.path.append('src')
    
    visualize(
        dataset_name='hai',
        net_name='hai_mlp',
        xp_path='log/hai_test',
        data_path='data/hai',
        load_model='log/hai_test/model.tar',
        device='cpu'
    )
