#!/usr/bin/env python3
"""
High Value Customer Prediction Application
Applies trained AI/OLS models to activation data
"""

import sys
import pandas as pd
import numpy as np
import os

def load_activation_data(file_path):
    """Load activation data from CSV"""
    try:
        data = pd.read_csv(file_path)
        print(f"✓ Loaded activation data: {data.shape}")
        print(f"  Columns: {list(data.columns)}")
        return data
    except Exception as e:
        print(f"✗ Error loading activation data: {e}")
        sys.exit(1)

def load_ai_model(model_path):
    """Load trained AI model"""
    try:
        from tensorflow import keras
        model = keras.models.load_model(model_path)
        print(f"✓ Loaded AI model from {model_path}")
        return model
    except Exception as e:
        print(f"✗ Error loading AI model: {e}")
        print(f"  Make sure TensorFlow is installed and model file exists")
        sys.exit(1)

def load_ols_model(model_path):
    """Load trained OLS model (sklearn LinearRegression saved via joblib/pickle)."""
    try:
        import joblib
        model = joblib.load(model_path)
        print(f"✓ Loaded OLS (sklearn) model from {model_path}")
        return model
    except Exception as e:
        print(f"✗ Error loading OLS model with joblib: {e}")
        try:
            import pickle
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            print(f"✓ Loaded OLS model with pickle fallback from {model_path}")
            return model
        except Exception as e2:
            print(f"✗ Error loading OLS model: {e2}")
            sys.exit(1)

def preprocess_data(data):
    """Preprocess activation data for prediction (force numeric float32)."""
    exclude_cols = ['customer_id', 'timestamp', 'label', 'id', 'target']
    feature_columns = [col for col in data.columns if col.lower() not in [c.lower() for c in exclude_cols]]

    print(f"✓ Using features: {feature_columns}")

    X_df = data[feature_columns].copy()

    # 1) Debug: show columns that are object dtype BEFORE conversion
    obj_cols = X_df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        print("⚠️ Found object dtype columns (will coerce to numeric):", obj_cols)
        # show a sample of problematic values
        for c in obj_cols[:5]:
            print(f"   Sample values in '{c}': {X_df[c].head(3).tolist()}")

    # 2) Convert everything to numeric; non-numeric -> NaN
    X_df = X_df.apply(pd.to_numeric, errors="coerce")

    # 3) Replace NaNs from coercion with 0 (or use median if you prefer)
    if X_df.isna().any().any():
        nan_cols = X_df.columns[X_df.isna().any()].tolist()
        print("⚠️ NaNs found after conversion in columns:", nan_cols)
        X_df = X_df.fillna(0)

    # 4) Convert to float32 numpy array (what Keras expects)
    X = X_df.to_numpy(dtype=np.float32)

    print(f"✓ Final input dtype: {X.dtype}, shape: {X.shape}")
    return X, feature_columns

def predict_with_ai(model, data):
    """Make predictions using AI model"""
    predictions = model.predict(data)
    # For binary classification
    predictions_binary = (predictions > 0.5).astype(int).flatten()
    return predictions, predictions_binary

def predict_with_ols(model, data):
    """Make predictions using sklearn LinearRegression model"""
    preds = model.predict(data).reshape(-1)
    preds_binary = (preds > 0.5).astype(int)
    return preds, preds_binary

def save_predictions(predictions, predictions_binary, output_path, model_type):
    """Save predictions to file"""
    df = pd.DataFrame({
        'prediction_probability': predictions.flatten(),
        'prediction_class': predictions_binary,
        'model_type': model_type
    })
    df.to_csv(output_path, index=False)
    print(f"✓ Saved {model_type} predictions to {output_path}")
    return df

def main():
    print("=" * 70)
    print("HIGH VALUE CUSTOMER PREDICTION SYSTEM")
    print("=" * 70)
    
    # Define paths
    activation_data_path = '/tmp/activationBase/activation_data.csv'
    ai_model_path = '/tmp/knowledgeBase/currentAiSolution.h5'
    ols_model_path = '/tmp/knowledgeBase/currentOlsSolution.pkl'
    output_dir = '/tmp/predictions'
    ai_output_path = f'{output_dir}/ai_predictions.csv'
    ols_output_path = f'{output_dir}/ols_predictions.csv'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check which mode to run
    mode = os.environ.get('PREDICTION_MODE', 'ai').lower()
    
    # Load activation data
    print("\n[1] Loading Activation Data...")
    print(f"    Path: {activation_data_path}")
    activation_data = load_activation_data(activation_data_path)
    
    # Preprocess data
    print("\n[2] Preprocessing Data...")
    X, feature_names = preprocess_data(activation_data)
    print(f"    Features shape: {X.shape}")
    
    if mode == 'ai':
        print("\n" + "=" * 70)
        print("RUNNING AI MODEL PREDICTION")
        print("=" * 70)
        
        # Load AI Model
        print("\n[3] Loading AI Model...")
        print(f"    Path: {ai_model_path}")
        ai_model = load_ai_model(ai_model_path)
        
        # Make Predictions
        print("\n[4] Making Predictions with AI Model...")
        ai_predictions, ai_predictions_binary = predict_with_ai(ai_model, X)
        
        # Save Predictions
        print("\n[5] Saving AI Predictions...")
        results_df = save_predictions(ai_predictions, ai_predictions_binary, 
                                      ai_output_path, 'AI_Neural_Network')
        
        # Print Summary
        print("\n" + "=" * 70)
        print("PREDICTION SUMMARY (AI Model)")
        print("=" * 70)
        print(f"Total customers analyzed: {len(ai_predictions_binary)}")
        print(f"High-value customers predicted: {np.sum(ai_predictions_binary)}")
        print(f"Regular customers predicted: {len(ai_predictions_binary) - np.sum(ai_predictions_binary)}")
        print(f"High-value percentage: {(np.sum(ai_predictions_binary)/len(ai_predictions_binary)*100):.2f}%")
        print(f"\nPredictions saved to: {ai_output_path}")
        
    elif mode == 'ols':
        print("\n" + "=" * 70)
        print("RUNNING OLS MODEL PREDICTION")
        print("=" * 70)
        
        # Load OLS Model
        print("\n[3] Loading OLS Model...")
        print(f"    Path: {ols_model_path}")
        ols_model = load_ols_model(ols_model_path)
        
        # Make Predictions
        print("\n[4] Making Predictions with OLS Model...")
        ols_predictions, ols_predictions_binary = predict_with_ols(ols_model, X)
        
        # Save Predictions
        print("\n[5] Saving OLS Predictions...")
        results_df = save_predictions(ols_predictions, ols_predictions_binary,
                                      ols_output_path, 'OLS_Regression')
        
        # Print Summary
        print("\n" + "=" * 70)
        print("PREDICTION SUMMARY (OLS Model)")
        print("=" * 70)
        print(f"Total customers analyzed: {len(ols_predictions_binary)}")
        print(f"High-value customers predicted: {np.sum(ols_predictions_binary)}")
        print(f"Regular customers predicted: {len(ols_predictions_binary) - np.sum(ols_predictions_binary)}")
        print(f"High-value percentage: {(np.sum(ols_predictions_binary)/len(ols_predictions_binary)*100):.2f}%")
        print(f"\nPredictions saved to: {ols_output_path}")
    
    print("\n" + "=" * 70)
    print("✓ PREDICTION PROCESS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()