#!/usr/bin/env python3
"""
AI Model Application for High Value Customer Prediction
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import sys

# Disable GPU (if not needed)
tf.config.set_visible_devices([], 'GPU')

# Define paths
ACTIVATION_DATA_PATH = "/tmp/activationBase/activation_data.csv"
MODEL_PATH = "/tmp/knowledgeBase/currentAiSolution.h5"
SCALER_X_PATH = "/tmp/knowledgeBase/scaler_X.pkl"
SCALER_Y_PATH = "/tmp/knowledgeBase/scaler_y.pkl"

def main():
    print("=" * 70)
    print("AI MODEL - HIGH VALUE CUSTOMER PREDICTION")
    print("=" * 70)
    
    # Load activation data
    print("\n[1] Loading activation data...")
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

    # --- Ensure all features are numeric (IMPORTANT for TensorFlow) ---
    print("\n[2.5] Enforcing numeric dtypes...")

    # Show which columns are problematic (object dtype)
    object_cols = activation_data.select_dtypes(include=["object"]).columns.tolist()
    if object_cols:
        print(f"⚠ Found object columns: {object_cols}")

    # Convert everything to numeric; non-convertible values become NaN
    activation_data = activation_data.apply(pd.to_numeric, errors="coerce")

    # Fill NaNs created by conversion
    nan_count = activation_data.isna().sum().sum()
    if nan_count > 0:
        print(f"⚠ Found {nan_count} NaN values after conversion; filling with 0")
    activation_data = activation_data.fillna(0)

    # Cast to float32 for TensorFlow
    activation_data = activation_data.astype(np.float32)
    print("✓ All activation features are numeric float32 now.")
    
    # Load scalers
    print("\n[2] Loading scalers...")
    try:
        scaler_X = joblib.load(SCALER_X_PATH)
        scaler_y = joblib.load(SCALER_Y_PATH)
        print("✓ Scalers loaded")
    except FileNotFoundError:
        print("⚠ Scalers not found, proceeding without scaling")
        scaler_X = None
        scaler_y = None
    
    # Preprocess data
    print("\n[3] Preprocessing data...")
    
    if scaler_X is not None:
        X_activation = scaler_X.transform(activation_data.to_numpy(dtype=np.float32))
    else:
        X_activation = activation_data.to_numpy(dtype=np.float32)

    print(f"✓ Activation input shape: {X_activation.shape}")
    
    # Load AI model
    print("\n[4] Loading AI model...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✓ AI model loaded")
        print(f"  Expected input shape: {model.input_shape}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        sys.exit(1)
    
    # Make predictions
    print("\n[5] Making predictions...")
    y_pred = model.predict(X_activation).flatten()
    
    # Inverse transform if scaler available
    if scaler_y is not None:
        pred_values = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    else:
        pred_values = y_pred
    
    # Convert to binary classification
    predictions_binary = (pred_values > 0.5).astype(int)
    
    # Prepare results
    results = activation_data.copy()
    results["prediction_probability"] = pred_values
    results["prediction_class"] = predictions_binary
    
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    print(f"\nTotal customers analyzed: {len(predictions_binary)}")
    print(f"High-value customers predicted: {np.sum(predictions_binary)}")
    print(f"Regular customers predicted: {len(predictions_binary) - np.sum(predictions_binary)}")
    print(f"High-value percentage: {(np.sum(predictions_binary)/len(predictions_binary)*100):.2f}%")
    
    print("\nFirst 5 predictions:")
    print(results[["prediction_probability", "prediction_class"]].head())
    
    print(f"\nFirst Predicted Value: {round(pred_values[0], 2)}")
    print(f"First Predicted Class: {predictions_binary[0]}")
    
    # Save results
    output_path = "/tmp/predictions/ai_predictions.csv"
    try:
        import os
        os.makedirs("/tmp/predictions", exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to: {output_path}")
    except Exception as e:
        print(f"\n⚠ Could not save results: {e}")
    
    print("\n" + "=" * 70)
    print("✓ AI PREDICTION COMPLETED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    main()