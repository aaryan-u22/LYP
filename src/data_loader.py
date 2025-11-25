import wfdb
import numpy as np
from scipy.signal import butter, filtfilt
from src.config import FS, WINDOW_SIZE

def denoise_signal(data):
    """Standard bandpass filter (0.5-50Hz)."""
    low, high = 0.5, 50.0
    nyq = 0.5 * FS
    b, a = butter(1, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data)

def load_and_segment_record(record_path):
    """
    Loads one patient file and segments it into windows.
    Returns: segments (list), labels (list)
    """
    try:
        # Load Raw
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal[:, 0] # Lead I
        annotation = wfdb.rdann(record_path, 'atr')
        
        clean_sig = denoise_signal(signal)
        
        segments = []
        labels = []
        
        # AAMI Mapping
        norm_sym = ['N', 'L', 'R', 'e', 'j']
        half_win = WINDOW_SIZE // 2
        
        for r, lbl in zip(annotation.sample, annotation.symbol):
            # Skip non-beat annotations
            if lbl in ['[', ']', '!', 'x', '|', '~', '+', '"', 'p', 't', 'u', '`', '\'', '^', 's', 'k', 'l']: 
                continue
            
            # Boundary Check
            if r - half_win < 0 or r + half_win >= len(clean_sig): 
                continue
            
            # Cut Window
            seg = clean_sig[r - half_win : r + half_win]
            
            # Artifact Rejection (>5mV is likely noise)
            if np.max(np.abs(seg)) > 5.0: 
                continue
            
            segments.append(seg)
            labels.append(0 if lbl in norm_sym else 1)
            
        return segments, labels
        
    except Exception as e:
        print(f"Error processing {record_path}: {e}")
        return [], []