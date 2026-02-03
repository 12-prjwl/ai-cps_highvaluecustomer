#!/usr/bin/env python3
"""
OLS Model Application for High Value Customer Prediction
"""

import numpy as np
import pandas as pd
from statsmodels.tools.tools import add_constant
import pickle
import joblib
import sys

# Define paths
ACTIVATION_DATA_PATH = "/tmp/activationBase/activation_data.csv"
MODEL_PATH = "/tmp/knowledgeBase/currentOlsSolution.pkl"

def main():
    print("=" * 70)
    print("OLS MODEL - HIGH VALUE CUSTOMER PREDICTION")
    print("=" * 70)
    
    # Load OLS model with multiple fallback methods
    print("\n[1] Loading OLS model...")
    ols_model = None
    
    try:
        # Method 1: Standard pickle
        with open(MODEL_PATH, "rb") as f:
            ols_model = pickle.load(f)
        print("✓ OLS model loaded successfully (standard pickle)")
    except Exception as e1:
        print(f"  Standard pickle failed: {e1}")
        
        try:
            # Method 2: Pickle with latin1 encoding (for Python 2 to 3 compatibility)
            with open(MODEL_PATH, "rb") as f:
                ols_model = pickle.load(f, encoding='latin1')
            print("✓ OLS model loaded successfully (latin1 encoding)")
        except Exception as e2:
            print(f"  Latin1 encoding failed: {e2}")
            
            try:
                # Method 3: Pickle with bytes encoding
                with open(MODEL_PATH, "rb") as f:
                    ols_model = pickle.load(f, encoding='bytes')
                print("✓ OLS model loaded successfully (bytes encoding)")
            except Exception as e3:
                print(f"  Bytes encoding failed: {e3}")
                
                try:
                    # Method 4: Try joblib (often more compatible)
                    ols_model = joblib.load(MODEL_PATH)
                    print("✓ OLS model loaded successfully (using joblib)")
                except Exception as e4:
                    print(f"  Joblib failed: {e4}")
                    print(f"\n✗ All loading methods failed!")
                    print(f"  Model path: {MODEL_PATH}")
                    print(f"\n  ERROR DETAILS:")
                    print(f"    - Standard pickle: {e1}")
                    print(f"    - Latin1 encoding: {e2}")
                    print(f"    - Bytes encoding: {e3}")
                    print(f"    - Joblib: {e4}")
                    print(f"\n  SOLUTION: Re-train and save the OLS model with current environment:")
                    print(f"    import joblib")
                    print(f"    joblib.dump(ols_model, 'currentOlsSolution.pkl')")
                    sys.exit(1)
    
    if ols_model is None:
        print("✗ Failed to load OLS model")
        sys.exit(1)
    
    # Load activation data
    print("\n[2] Loading activation data...")
    try:
        activation_data = pd.read_csv(ACTIVATION_DATA_PATH)
        print(f"✓ Loaded activation data: {activation_data.shape}")
        print(f"  Columns: {list(activation_data.columns)}")
    except Exception as e:
        print(f"✗ Error loading activation data: {e}")
        sys.exit(1)
    
    # Remove target column if exists
    if "high_value" in activation_data.columns:
        activation_data = activation_data.drop(columns=["high_value"])
    if "label" in activation_data.columns:
        activation_data = activation_data.drop(columns=["label"])

    # --- Ensure all features are numeric (IMPORTANT for Statsmodels) ---
    print("\n[2.5] Enforcing numeric dtypes...")

    object_cols = activation_data.select_dtypes(include=["object"]).columns.tolist()
    if object_cols:
        print(f"⚠ Found object columns: {object_cols}")

    # Convert all columns to numeric; non-convertible values become NaN
    activation_data = activation_data.apply(pd.to_numeric, errors="coerce")

    # Fill NaNs created by conversion
    nan_count = activation_data.isna().sum().sum()
    if nan_count > 0:
        print(f"⚠ Found {nan_count} NaN values after conversion; filling with 0")
    activation_data = activation_data.fillna(0)

    # Cast to float for statsmodels
    activation_data = activation_data.astype(float)
    print("✓ All activation features are numeric float now.")

    
    # Add constant for OLS
    print("\n[3] Preprocessing data...")
    
    # Check if model expects a constant term
    # Try to detect from model parameters
    model_params = None
    has_const_in_model = False
    
    try:
        if hasattr(ols_model, 'params'):
            model_params = ols_model.params
            param_names = list(model_params.index) if hasattr(model_params, 'index') else []
            has_const_in_model = 'const' in param_names
            print(f"  Model parameter names: {param_names}")
            print(f"  Model has constant term: {has_const_in_model}")
    except:
        print("  Could not detect model parameters")
    
    # Only add constant if the model was trained with one
    if has_const_in_model:
        print("  Adding constant to activation data...")
        activation_data_processed = add_constant(activation_data, has_constant="add")
    else:
        print("  Model was trained WITHOUT constant - skipping add_constant()")
        activation_data_processed = activation_data.copy()
    
    print(f"✓ Activation data shape after preprocessing: {activation_data_processed.shape}")
    print(f"  Columns: {list(activation_data_processed.columns)}")
    
    # Make predictions
    print("\n[4] Making predictions...")
    try:
        predictions = ols_model.predict(activation_data_processed)
    except Exception as e:
        print(f"✗ Error making predictions: {e}")
        print(f"  This might be due to feature mismatch between training and activation data")
        
        # Print diagnostic info
        if hasattr(ols_model, 'params'):
            print(f"  Model features: {list(ols_model.params.index)}")
        print(f"  Activation features: {list(activation_data_processed.columns)}")
        
        # Try alternative: match exact feature order from model
        print("\n  Attempting to reorder features to match model...")
        try:
            if hasattr(ols_model, 'params'):
                expected_features = list(ols_model.params.index)
                # Reorder or select only the features the model expects
                activation_data_processed = activation_data_processed[expected_features]
                predictions = ols_model.predict(activation_data_processed)
                print("  ✓ Prediction successful after feature reordering")
            else:
                raise Exception("Cannot determine expected features from model")
        except Exception as e2:
            print(f"  ✗ Feature reordering failed: {e2}")
            sys.exit(1)
    
    # Convert to binary classification
    predictions_binary = (predictions > 0.5).astype(int)
    
    # Prepare results (use original activation_data without const column for output)
    output = activation_data.copy()
    output["prediction_probability"] = predictions
    output["prediction_class"] = predictions_binary
    
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    print(f"\nTotal customers analyzed: {len(predictions_binary)}")
    print(f"High-value customers predicted: {np.sum(predictions_binary)}")
    print(f"Regular customers predicted: {len(predictions_binary) - np.sum(predictions_binary)}")
    print(f"High-value percentage: {(np.sum(predictions_binary)/len(predictions_binary)*100):.2f}%")
    
    print("\nFirst 5 predictions:")
    print(output[["prediction_probability", "prediction_class"]].head())
    
    print(f"\nFirst Predicted Value: {round(predictions[0], 2)}")
    print(f"First Predicted Class: {predictions_binary[0]}")
    
    # Save results
    output_path = "/tmp/predictions/ols_predictions.csv"
    try:
        import os
        os.makedirs("/tmp/predictions", exist_ok=True)
        output.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to: {output_path}")
    except Exception as e:
        print(f"\n⚠ Could not save results: {e}")
    
    print("\n" + "=" * 70)
    print("✓ OLS PREDICTION COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()