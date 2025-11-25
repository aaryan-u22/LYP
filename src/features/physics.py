import numpy as np
import nolds
import antropy as ant
from scipy.spatial.distance import pdist, squareform
import warnings

# Suppress specific library warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def calculate_rqa_metrics(signal, dim=3, tau=2, threshold_factor=0.1):
    """
    Recurrence Quantification Analysis (RQA).
    Extracts structural rules from the phase space trajectory.
    """
    N = len(signal)
    if N < dim * tau: return [0., 0., 0.]
    
    # 1. Phase Space Reconstruction
    phase_space = np.array([signal[i: i + dim*tau : tau] for i in range(N - (dim-1)*tau)])
    
    # 2. Distance Matrix
    dists = pdist(phase_space, metric='euclidean')
    
    # 3. Thresholding
    threshold = threshold_factor * np.std(signal)
    rec_plot = (squareform(dists) <= threshold).astype(int)
    
    if rec_plot.shape[0] == 0: return [0., 0., 0.]
    
    # 4. Extract Metrics
    # RR: Recurrence Rate (Density)
    rr = np.sum(rec_plot) / (rec_plot.shape[0] ** 2)
    
    # DET: Determinism (Diagonal structures -> Predictability)
    diagonals = [np.trace(rec_plot, offset=k) for k in range(1, rec_plot.shape[0])]
    det = np.mean([d for d in diagonals if d > 2]) if any(d > 2 for d in diagonals) else 0
    
    # LAM: Laminarity (Vertical structures -> Stagnation)
    verticals = np.sum(rec_plot, axis=0)
    lam = np.mean(verticals[verticals > 2]) if any(verticals > 2) else 0
    
    return [rr, det, lam]

def extract_chaos_features(segment):
    """
    Extracts the Chaos Vector: [LLE, FD, SampEn, RR, DET, LAM]
    """
    try:
        # CRITICAL FIX: Enforce memory layout for Numba/Antropy stability
        segment = np.ascontiguousarray(segment, dtype=np.float64)
        
        # 1. Chaos (Stability)
        lle = nolds.lyap_r(segment, emb_dim=3, lag=1, min_tsep=None)
        
        # 2. Complexity (Fractal Dimension)
        fd = ant.higuchi_fd(segment, kmax=10)
        
        # 3. Regularity (Entropy)
        sampen = ant.sample_entropy(segment)
        
        # 4. Structure (RQA)
        rqa = calculate_rqa_metrics(segment)
        
        features = [lle, fd, sampen] + rqa
        
        # Sanity Check: Replace NaNs/Infs with 0.0
        return [0.0 if np.isnan(x) or np.isinf(x) else x for x in features]
        
    except Exception:
        # Fail-safe for mathematical crashes
        return [0.0] * 6