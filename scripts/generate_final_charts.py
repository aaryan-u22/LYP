import matplotlib.pyplot as plt
import numpy as np
import os

SAVE_DIR = 'models/final_charts/'
os.makedirs(SAVE_DIR, exist_ok=True)

def plot_victory_chart():
    # 1. THE DATA (Complete Tournament Results)
    # -------------------------------------------------
    models = [
        'Traditional\n(Random Forest)', 
        'Modern\n(1D-CNN)', 
        'Proposed: LogReg\n(Linear)', 
        'Proposed: SVM\n(Geometric)', 
        'Proposed: XGBoost\n(Tree-Based)'
    ]
    
    # Accuracy (Overall Performance)
    accuracy = [92.00, 94.88, 79.42, 95.25, 97.47]
    
    # F1-Score for Anomalies (The "True" Diagnostic Power)
    # These numbers come from your previous run logs
    f1_scores = [80.0, 84.0, 56.0, 87.0, 93.0] 
    
    # 2. PLOT SETUP
    # -------------------------------------------------
    x = np.arange(len(models))
    width = 0.35 

    # Use a professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 7)) # Wider to fit 5 models
    
    # 3. CREATE BARS
    rects1 = ax.bar(x - width/2, accuracy, width, label='Accuracy', 
                    color='#2c3e50', edgecolor='white', linewidth=1)
    
    rects2 = ax.bar(x + width/2, f1_scores, width, label='Anomaly F1-Score', 
                    color='#e74c3c', edgecolor='white', linewidth=1)

    # 4. STYLING & LABELS
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('COMPARISON CHART', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax.set_ylim(50, 105) # Lower floor to 50 to show LogReg failure
    
    # Legend
    ax.legend(loc='lower right', frameon=True, framealpha=0.9, shadow=True, fontsize=11)
    
    # Highlight the Winner (XGB)
    winner_idx = 4
    ax.text(winner_idx, 102, "BEST", ha='center', va='bottom', fontweight='bold', color='#27ae60', fontsize=12)
    
    # 5. ADD VALUE LABELS
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    add_labels(rects1)
    add_labels(rects2)

    # 6. SAVE
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'final_tournament_full.png')
    plt.savefig(save_path, dpi=300)
    print(f"Chart saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_victory_chart()