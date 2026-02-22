import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import torch
import torch.nn.functional as F
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def extract_temporal_attention(attention_matrix):
    if attention_matrix.dim() == 3:
        attention_matrix = attention_matrix.mean(dim=0)
    
    temporal_attention = attention_matrix.mean(dim=0)
    temporal_attention = (temporal_attention - temporal_attention.min()) / \
                        (temporal_attention.max() - temporal_attention.min() + 1e-8)
    
    return temporal_attention.cpu().numpy()


def visualize_multi_sample_attention(model, X_test, y_test, device, n_samples=6,
                                     save_path='multi_sample_attention_fixed.png'):
    model.eval()
    
    normal_indices = np.where(y_test.cpu().numpy() == 0)[0]
    abnormal_indices = np.where(y_test.cpu().numpy() == 1)[0]
    
    n_normal = min(n_samples // 2, len(normal_indices))
    n_abnormal = min(n_samples - n_normal, len(abnormal_indices))
    
    selected_indices = list(np.random.choice(normal_indices, n_normal, replace=False)) + \
                      list(np.random.choice(abnormal_indices, n_abnormal, replace=False))
    
    fig, axes = plt.subplots(len(selected_indices), 2, figsize=(16, len(selected_indices)*3))
    
    if len(selected_indices) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sample_idx in enumerate(selected_indices):
        if isinstance(X_test, torch.Tensor):
            signal = X_test[sample_idx].cpu().numpy()
        else:
            signal = X_test[sample_idx]
        
        if signal.ndim > 1:
            signal = signal.squeeze()
        
        label = y_test[sample_idx].item() if isinstance(y_test, torch.Tensor) else y_test[sample_idx]
        
        with torch.no_grad():
            if isinstance(X_test, torch.Tensor):
                input_tensor = X_test[sample_idx].unsqueeze(0).to(device)
                if input_tensor.dim() == 2:
                    input_tensor = input_tensor.unsqueeze(1)
            else:
                input_tensor = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(0).to(device)
            
            output = model(input_tensor)
            
            if output.shape[1] == 2:
                prediction = torch.argmax(output, dim=1).item()
                confidence = F.softmax(output, dim=1)[0, prediction].item()
            else:
                prediction = (torch.sigmoid(output) > 0.5).int().item()
                confidence = torch.sigmoid(output).item() if prediction == 1 else (1 - torch.sigmoid(output).item())
            
            attention_matrix = model.get_attention_map()
        
        if attention_matrix is None or torch.all(attention_matrix == 0):
            print(f"Warning: Sample {idx+1} has zero/None attention weights")
            temporal_attn = np.zeros(len(signal))
        else:
            temporal_attn = extract_temporal_attention(attention_matrix)
            
            if len(temporal_attn) != len(signal):
                x_old = np.linspace(0, 1, len(temporal_attn))
                x_new = np.linspace(0, 1, len(signal))
                f = interp1d(x_old, temporal_attn, kind='cubic', fill_value='extrapolate')
                temporal_attn = f(x_new)
                temporal_attn = np.clip(temporal_attn, 0, 1)
        
        time_axis = np.arange(len(signal)) / 360
        
        axes[idx, 0].plot(time_axis, signal, 'b-', linewidth=1.5)
        axes[idx, 0].set_ylabel('Amplitude')
        axes[idx, 0].set_title(
            f'Sample {idx+1}: True={"Normal" if label==0 else "Abnormal"}, '
            f'Pred={"Normal" if prediction==0 else "Abnormal"}, '
            f'Conf={confidence*100:.1f}%',
            fontweight='bold'
        )
        axes[idx, 0].grid(True, alpha=0.3)
        
        if np.max(temporal_attn) > 0:
            color = 'coral' if label == 0 else 'steelblue'
            axes[idx, 1].fill_between(time_axis, 0, temporal_attn, alpha=0.6, color=color)
            axes[idx, 1].plot(time_axis, temporal_attn, 
                             'r-' if label == 0 else 'b-', linewidth=2)
            axes[idx, 1].set_ylabel('Attention Weight')
            axes[idx, 1].set_ylim([0, 1.05])
            axes[idx, 1].set_title(
                f'{"Normal" if label==0 else "Abnormal"} - Attention Pattern',
                fontweight='bold'
            )
        else:
            axes[idx, 1].text(0.5, 0.5, 'No Attention Data', 
                             ha='center', va='center', 
                             transform=axes[idx, 1].transAxes, fontsize=12)
            axes[idx, 1].set_title('Attention Not Available', fontweight='bold')
        
        axes[idx, 1].grid(True, alpha=0.3)
        
        if idx == len(selected_indices) - 1:
            axes[idx, 0].set_xlabel('Time (s)')
            axes[idx, 1].set_xlabel('Time (s)')
    
    plt.suptitle('Attention Patterns: Normal vs Abnormal ECG Beats', 
                fontsize=16, fontweight='bold', y=1.001)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {save_path}")
    return fig


def visualize_attention_comprehensive(model, X_test, y_test, device, sample_idx=0,
                                     sampling_rate=360,
                                     save_path='attention_comprehensive.png'):
    model.eval()
    
    if isinstance(X_test, torch.Tensor):
        signal = X_test[sample_idx].cpu().numpy().squeeze()
    else:
        signal = X_test[sample_idx].squeeze()
    
    true_label = y_test[sample_idx].item() if isinstance(y_test, torch.Tensor) else y_test[sample_idx]
    
    with torch.no_grad():
        if isinstance(X_test, torch.Tensor):
            input_tensor = X_test[sample_idx].unsqueeze(0).to(device)
            if input_tensor.dim() == 2:
                input_tensor = input_tensor.unsqueeze(1)
        else:
            input_tensor = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(0).to(device)
        
        output = model(input_tensor)
        
        if output.shape[1] == 2:
            pred_label = torch.argmax(output, dim=1).item()
            confidence = F.softmax(output, dim=1)[0, pred_label].item()
        else:
            pred_label = (torch.sigmoid(output) > 0.5).int().item()
            confidence = torch.sigmoid(output).item() if pred_label == 1 else (1 - torch.sigmoid(output).item())
        
        attention_matrix = model.get_attention_map()
    
    if attention_matrix is not None and not torch.all(attention_matrix == 0):
        temporal_attn = extract_temporal_attention(attention_matrix)
        
        if len(temporal_attn) != len(signal):
            x_old = np.linspace(0, 1, len(temporal_attn))
            x_new = np.linspace(0, 1, len(signal))
            f = interp1d(x_old, temporal_attn, kind='cubic', fill_value='extrapolate')
            temporal_attn = f(x_new)
            temporal_attn = np.clip(temporal_attn, 0, 1)
    else:
        temporal_attn = np.zeros(len(signal))
        print("Warning: No valid attention weights found")
    
    time_axis = np.arange(len(signal)) / sampling_rate
    peaks, _ = find_peaks(signal, distance=sampling_rate//3, prominence=0.3)
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_axis, signal, 'b-', linewidth=1.5, label='ECG Signal')
    if len(peaks) > 0:
        ax1.plot(time_axis[peaks], signal[peaks], 'ro', markersize=8, 
                label=f'R-peaks (n={len(peaks)})')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude (normalized)')
    ax1.set_title(
        f'ECG Signal | True: {"Normal" if true_label==0 else "Abnormal"}, '
        f'Predicted: {"Normal" if pred_label==0 else "Abnormal"}, '
        f'Confidence: {confidence*100:.1f}%',
        fontweight='bold', fontsize=13
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    ax2 = fig.add_subplot(gs[1, 0])
    if np.max(temporal_attn) > 0:
        for i in range(len(time_axis)-1):
            ax2.axvspan(time_axis[i], time_axis[i+1], 
                       alpha=temporal_attn[i]*0.5, color='red')
    ax2.plot(time_axis, signal, 'k-', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Signal with Attention Overlay (Red = High Attention)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, 1])
    if np.max(temporal_attn) > 0:
        ax3.fill_between(time_axis, 0, temporal_attn, alpha=0.6, color='coral')
        ax3.plot(time_axis, temporal_attn, 'r-', linewidth=2)
        
        peak_idx = np.argmax(temporal_attn)
        peak_time = time_axis[peak_idx]
        ax3.axvline(peak_time, color='darkred', linestyle='--', linewidth=2,
                   label=f'Peak @ {peak_time:.2f}s')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No Attention Data', ha='center', va='center',
                transform=ax3.transAxes, fontsize=12)
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Attention Weight')
    ax3.set_title('Temporal Attention Weights', fontweight='bold')
    ax3.set_ylim([0, 1.05])
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig.add_subplot(gs[2, :])
    if attention_matrix is not None and not torch.all(attention_matrix == 0):
        attn_viz = attention_matrix[0].cpu().numpy() if attention_matrix.dim() == 3 else attention_matrix.cpu().numpy()
        
        if attn_viz.shape[0] > 100:
            step = attn_viz.shape[0] // 100
            attn_viz = attn_viz[::step, ::step]
        
        im = ax4.imshow(attn_viz, cmap='hot', aspect='auto', interpolation='bilinear')
        ax4.set_xlabel('Key Position (time)')
        ax4.set_ylabel('Query Position (time)')
        ax4.set_title('Self-Attention Matrix', fontweight='bold')
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20)
    else:
        ax4.text(0.5, 0.5, 'No Attention Matrix', ha='center', va='center',
                transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Attention Matrix Not Available', fontweight='bold')
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {save_path}")
    return fig


if __name__ == "__main__":
    print("ECG-CANET Visualization Module")
    print("Usage: import and call visualize_multi_sample_attention(model, X_test, y_test, device)")