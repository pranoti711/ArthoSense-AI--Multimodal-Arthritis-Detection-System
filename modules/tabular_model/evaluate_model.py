
import os
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt  # For ROC curve visualization

def load_and_evaluate(show_plot=False):
    processed_dir = "data/processed/labs_clinical/arthritis_profile_processed"
    
    try:
        df = pd.read_parquet(os.path.join(processed_dir, "multimodal.parquet"))  # Load your processed data
        model = joblib.load(os.path.join(processed_dir, "trained_model.pkl"))  # Load the saved model
        print(f"Loaded data: {df.shape} rows/cols. Model loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}. Please ensure preprocessing and training have run.")
        return
    
    features = ['ESRh', 'ESRo', 'CRP', 'Uric_Acid', 'ASO', 'age_binned', 'sex_encoded']
    target = 'target_encoded'
    
    X = df[features]  
    y = df[target]    
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
  
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    if y_prob is not None:
        auc_score = roc_auc_score(y_test, y_prob)
        print(f"AUC-ROC Score: {auc_score:.2f}")
        
       
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')  
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='best')
        plt.savefig(os.path.join(processed_dir, "roc_curve.png"))  
        plt.show() 
        plt.close()  
        print("Evaluation complete. ROC curve saved as roc_curve.png if not shown.")
    else:
        print("No probabilities available for AUC-ROC.")

if __name__ == "__main__":
    print("Starting model evaluation...")
    load_and_evaluate(show_plot=True)  

