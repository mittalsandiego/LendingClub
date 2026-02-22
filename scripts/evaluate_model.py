"""
scripts/evaluate_model.py
Checkpoint 3 — Evaluation Gate

Downloads the trained model artifact from S3, loads it locally,
runs inference on the held-out test set, and computes:
  - Overall accuracy
  - Per-class precision, recall, F1
  - Weighted F1 score
  - Confusion matrix

Fails (exit 1) if accuracy < threshold OR weighted F1 < threshold,
blocking model registration in Checkpoint 4.

Usage:
    python scripts/evaluate_model.py \
        --bucket sagemaker-us-east-1-099405935674 \
        --prefix aai540-group9-loan-project \
        --model-artifact-uri s3://.../model.tar.gz \
        --accuracy-threshold 0.75 \
        --f1-threshold 0.70 \
        --output-file evaluation_report.json
"""

import argparse
import json
import os
import sys
import tarfile
import tempfile
from datetime import datetime

import boto3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Class names match the label encoding in the notebook:
# 0=Fully Paid, 1=Charged Off, 2=Current
CLASS_NAMES = ['Fully Paid', 'Charged Off', 'Current']


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained XGBoost model')
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--model-artifact-uri', required=True)
    parser.add_argument('--accuracy-threshold', type=float, default=0.75)
    parser.add_argument('--f1-threshold', type=float, default=0.70)
    parser.add_argument('--output-file', default='evaluation_report.json')
    return parser.parse_args()


def download_and_extract_model(model_artifact_uri, tmp_dir):
    """Download model.tar.gz from S3 and extract the XGBoost model file."""
    s3 = boto3.client('s3')

    # Parse s3://bucket/key
    uri_parts = model_artifact_uri.replace('s3://', '').split('/', 1)
    bucket, key = uri_parts[0], uri_parts[1]

    local_tar = os.path.join(tmp_dir, 'model.tar.gz')
    print(f"  Downloading model artifact from {model_artifact_uri}...")
    s3.download_file(bucket, key, local_tar)

    with tarfile.open(local_tar, 'r:gz') as tar:
        tar.extractall(tmp_dir)

    # Find the XGBoost model file
    for fname in os.listdir(tmp_dir):
        if fname in ('xgboost-model', 'model') or fname.endswith('.json') or fname.endswith('.ubj'):
            return os.path.join(tmp_dir, fname)

    raise FileNotFoundError(f"Could not find XGBoost model file in {tmp_dir}: {os.listdir(tmp_dir)}")


def load_test_data(bucket, prefix):
    """Download the held-out test CSV from S3."""
    s3 = boto3.client('s3')
    test_key = f"{prefix}/test/test.csv"

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        tmp_path = f.name

    print(f"  Downloading test data from s3://{bucket}/{test_key}...")
    s3.download_file(bucket, test_key, tmp_path)

    # Label is the first column (SageMaker XGBoost convention)
    df = pd.read_csv(tmp_path, header=None)
    y = df.iloc[:, 0].values.astype(int)
    X = df.iloc[:, 1:].values.astype(float)

    os.unlink(tmp_path)
    return X, y


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("CHECKPOINT 3: EVALUATION GATE")
    print("="*60)
    print(f"  Accuracy threshold:    >= {args.accuracy_threshold}")
    print(f"  Weighted F1 threshold: >= {args.f1_threshold}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Download and load model
        print("\n[1/3] Loading model...")
        model_path = download_and_extract_model(args.model_artifact_uri, tmp_dir)
        booster = xgb.Booster()
        booster.load_model(model_path)
        print(f"  Model loaded from {model_path}")

        # Load test data
        print("\n[2/3] Loading test data...")
        X_test, y_test = load_test_data(args.bucket, args.prefix)
        print(f"  Test set: {len(y_test):,} records, {X_test.shape[1]} features")

        # Run inference
        print("\n[3/3] Running evaluation...")
        dtest = xgb.DMatrix(X_test)
        probs = booster.predict(dtest)          # shape: (n, 3)
        y_pred = np.argmax(probs, axis=1)

    # Compute metrics
    accuracy    = float(accuracy_score(y_test, y_pred))
    weighted_f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
    macro_f1    = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
    cm          = confusion_matrix(y_test, y_pred).tolist()
    report_str  = classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0)

    print(f"\n{report_str}")
    print(f"  Accuracy:    {accuracy:.4f}  (threshold >= {args.accuracy_threshold})")
    print(f"  Weighted F1: {weighted_f1:.4f}  (threshold >= {args.f1_threshold})")
    print(f"  Macro F1:    {macro_f1:.4f}")

    # Gate logic
    accuracy_passed    = accuracy    >= args.accuracy_threshold
    f1_passed          = weighted_f1 >= args.f1_threshold
    evaluation_passed  = accuracy_passed and f1_passed

    print(f"\n  Accuracy gate:    {'PASS' if accuracy_passed    else ' FAIL'}")
    print(f"  Weighted F1 gate: {'PASS' if f1_passed          else ' FAIL'}")
    print(f"  Overall gate:     {'PASS' if evaluation_passed  else ' FAIL'}")

    # Build evaluation report
    report = {
        'timestamp':          datetime.utcnow().isoformat(),
        'accuracy':           accuracy,
        'weighted_f1':        weighted_f1,
        'macro_f1':           macro_f1,
        'confusion_matrix':   cm,
        'class_names':        CLASS_NAMES,
        'test_set_size':      len(y_test),
        'accuracy_threshold': args.accuracy_threshold,
        'f1_threshold':       args.f1_threshold,
        'accuracy_passed':    accuracy_passed,
        'f1_passed':          f1_passed,
        'evaluation_passed':  evaluation_passed,
        'classification_report': report_str,
    }

    with open(args.output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n  Report written to {args.output_file}")
    print("\n" + "="*60)

    if not evaluation_passed:
        print(" CHECKPOINT 3 FAILED — model does not meet quality thresholds")
        print("   Model will NOT be registered. Review training data and hyperparameters.")
        print("="*60 + "\n")
        sys.exit(1)

    print("CHECKPOINT 3 PASSED — Evaluation gate cleared")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
