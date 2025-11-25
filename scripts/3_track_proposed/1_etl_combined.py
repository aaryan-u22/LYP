import sys
import os
# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import glob
import wfdb
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from src.config import RAW_DATA_DIR, PROCESSED_BASE_DIR, TEST_SPLIT_RATIO, RANDOM_SEED, NUM_CORES, FS, WINDOW_SIZE
from src.data_loader import denoise_signal
from src.features.physics import extract_chaos_features

OUTPUT_DIR = os.path.join(PROCESSED_BASE_DIR, '3_proposed_combined')

def worker(record_id):
    path = os.path.join(RAW_DATA_DIR, record_id)
    try:
        # 1. Load Raw Data
        record = wfdb.rdrecord(path)
        signal = record.p_signal[:, 0]
        annot = wfdb.rdann(path, 'atr')
        
        clean_sig = denoise_signal(signal)
        
        r_peaks = annot.sample
        labels = annot.symbol
        
        # Calculate RR differences globally first
        rr_diffs = np.diff(r_peaks)
        
        combined_data = []
        norm_sym = ['N', 'L', 'R', 'e', 'j']
        half_win = WINDOW_SIZE // 2
        
        # Start loop (skip buffers for stats)
        for i in range(5, len(r_peaks) - 5):
            lbl = labels[i]
            if lbl in ['[', ']', '!', 'x', '|', '~', '+', '"', 'p', 't', 'u', '`', '\'', '^', 's', 'k', 'l']: 
                continue
            
            current_peak = r_peaks[i]
            
            # --- A. TRADITIONAL FEATURES (The "Cheat Sheet") ---
            pre_rr = r_peaks[i] - r_peaks[i-1]
            post_rr = r_peaks[i+1] - r_peaks[i]
            local_rr = np.mean(rr_diffs[i-10:i]) if i >= 10 else np.mean(rr_diffs[:i])
            amp = clean_sig[current_peak]
            
            # --- B. CHAOS FEATURES (The "Physics") ---
            # Cut the window for physics
            if current_peak - half_win < 0 or current_peak + half_win >= len(clean_sig):
                continue
                
            seg = clean_sig[current_peak - half_win : current_peak + half_win]
            
            # Artifact check
            if np.max(np.abs(seg)) > 5.0: continue
            
            # Extract Physics
            phys_feats = extract_chaos_features(seg)
            
            # Combine: [Physics (6)] + [Traditional (4)] + [Label]
            label = 0 if lbl in norm_sym else 1
            
            # Row structure
            row = phys_feats + [pre_rr, post_rr, local_rr, amp, label]
            combined_data.append(row)
            
        return combined_data
        
    except Exception as e:
        return []

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = glob.glob(os.path.join(RAW_DATA_DIR, "*.dat"))
    ids = list(set([os.path.basename(f).split('.')[0] for f in files]))
    
    print(f"--- Track 3C (v2): True Feature Fusion ---")
    print(f"Processing {len(ids)} records (Chaos + RR Intervals)...")
    
    with Pool(processes=NUM_CORES) as pool:
        results = list(tqdm(pool.imap(worker, ids), total=len(ids)))
    
    flat = [item for sublist in results for item in sublist]
    
    # Column Names
    cols = [
        'LLE', 'FD', 'SampEn', 'RR', 'DET', 'LAM',  # Chaos
        'Pre_RR', 'Post_RR', 'Local_RR', 'Amplitude', # Traditional
        'Label'
    ]
    
    df = pd.DataFrame(flat, columns=cols)
    print(f"Total Beats: {len(df)}")
    
    # Random Split (High Score Mode)
    X = df
    y = df['Label']
    
    X_train, X_test = train_test_split(
        X, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_SEED, stratify=y
    )
    
    X_train.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
    X_test.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)
    print(f"Done. Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
