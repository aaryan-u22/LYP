import numpy as np

def get_rr_interval_features(r_peaks, signal):
    """
    Extracts traditional timing statistics (Track 1).
    Returns list of [Pre_RR, Post_RR, Local_Avg_RR, Amplitude]
    """
    features = []
    valid_indices = []
    
    # Calculate R-R intervals (diff between peaks)
    rr_diffs = np.diff(r_peaks)
    
    if len(r_peaks) < 15: return [], []

    for i in range(5, len(r_peaks) - 5):
        try:
            current_idx = r_peaks[i]
            
            # 1. Timing Features
            pre_rr = r_peaks[i] - r_peaks[i-1]
            post_rr = r_peaks[i+1] - r_peaks[i]
            
            # Local Avg (Last 10 beats)
            local_rr = np.mean(rr_diffs[i-10:i]) if i >= 10 else np.mean(rr_diffs[:i])
            
            # 2. Morphological Feature (Simple Amplitude)
            if current_idx < len(signal):
                amp = signal[current_idx]
            else:
                amp = 0
            
            features.append([pre_rr, post_rr, local_rr, amp])
            valid_indices.append(i)
            
        except IndexError:
            continue
            
    return features, valid_indices