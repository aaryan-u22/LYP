import sys
import os
# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import pandas as pd
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from src.config import PROCESSED_BASE_DIR, MODEL_BASE_DIR, RANDOM_SEED

# --- CONFIG ---
DATA_DIR = os.path.join(PROCESSED_BASE_DIR, '3_proposed_combined')
SAVE_DIR = os.path.join(MODEL_BASE_DIR, '3_proposed_xgb')

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print(f"--- Track 3C: Training XGBoost (Combined Features) ---")
    print(f"Loading Data from {DATA_DIR}...")
    try:
        train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
        test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    except FileNotFoundError:
        print("Error: Combined data not found. Run 1_etl_combined_features.py first.")
        return
    
    # Prepare X/y
    X_train = train.drop(['Label', 'PatientID'], axis=1, errors='ignore')
    y_train = train['Label']
    X_test = test.drop(['Label', 'PatientID'], axis=1, errors='ignore')
    y_test = test['Label']
    
    # 1. Scale
    print("Scaling Features...")
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # 2. SMOTE
    print("Applying SMOTE...")
    smote = SMOTE(random_state=RANDOM_SEED)
    X_res, y_res = smote.fit_resample(X_train_s, y_train)
    
    # 3. Train
    print("Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(X_res, y_res)
    
    # 4. Evaluate
    print("\n--- RESULTS (Track 3C: Combined) ---")
    y_pred = model.predict(X_test_s)
    
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    
    print(f"Final Accuracy:       {acc*100:.2f}%")
    print(f"Sensitivity (Recall): {sens:.4f}")
    print(f"Specificity:          {spec:.4f}")
    
    # 5. Explainability (SHAP) with JSON Workaround
    print("\nGenerating SHAP Plot...")
    try:
        # --- FIX START ---
        # 1. Save model to a temporary JSON file. This strips scikit-learn wrappers.
        temp_model_path = os.path.join(SAVE_DIR, 'temp_xgb.json')
        model.save_model(temp_model_path)
        
        # 2. Load it back as a raw Booster
        booster = xgb.Booster()
        booster.load_model(temp_model_path)
        
        # 3. Use the CLEAN booster for SHAP
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X_test_s.iloc[:1000])
        
        # 4. Plot
        plt.figure()
        shap.summary_plot(shap_values, X_test_s.iloc[:1000], show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'shap_summary.png'), bbox_inches='tight')
        print("SHAP plot saved successfully.")
        
        # Cleanup
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
        # --- FIX END ---
        
    except Exception as e:
        print(f"SHAP Error: {e}")
    
    joblib.dump(model, os.path.join(SAVE_DIR, 'xgboost_combined.pkl'))
    joblib.dump(scaler, os.path.join(SAVE_DIR, 'scaler.pkl'))
    print(f"Model saved to {SAVE_DIR}")

if __name__ == "__main__":
    main()