"""
scripts/train_model.py
Checkpoint 2 — SageMaker Training Job

Launches a managed SageMaker XGBoost training job using the
hyperparameters defined in the ML System Design Document.
Waits for completion and writes the model artifact URI to
an output JSON file for downstream pipeline steps.

Usage:
    python scripts/train_model.py \
        --bucket sagemaker-us-east-1-099405935674 \
        --prefix aai540-group9-loan-project \
        --instance-type ml.m5.xlarge \
        --output-file training_outputs.json
"""

import argparse
import json
import os
import sys
from datetime import datetime

import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput


# XGBoost hyperparameters — matches Section 5 of the ML Design Document
# and the actual training run (sagemaker-xgboost-2026-02-21-12-22-16-379)
HYPERPARAMETERS = {
    'objective':             'multi:softprob',
    'num_class':             '3',
    'num_round':             '150',
    'max_depth':             '6',
    'eta':                   '0.1',
    'subsample':             '0.8',
    'colsample_bytree':      '0.8',
    'eval_metric':           'merror',
    'early_stopping_rounds': '10',
}

XGBOOST_VERSION = '1.7-1'


def parse_args():
    parser = argparse.ArgumentParser(description='Launch SageMaker XGBoost training job')
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--instance-type', default='ml.m5.xlarge')
    parser.add_argument('--output-file', default='training_outputs.json')
    return parser.parse_args()


def main():
    args = parse_args()

    region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
    role = os.environ.get('SAGEMAKER_ROLE', sagemaker.get_execution_role())

    sess = sagemaker.Session()
    run_id = datetime.utcnow().strftime('%Y-%m-%d-%H%M%S')

    print("\n" + "="*60)
    print("CHECKPOINT 2: SAGEMAKER TRAINING JOB")
    print("="*60)
    print(f"  Instance:    {args.instance_type}")
    print(f"  XGBoost:     {XGBOOST_VERSION}")
    print(f"  Bucket:      s3://{args.bucket}/{args.prefix}")
    print(f"  Run ID:      {run_id}")

    # Resolve training data S3 paths
    train_s3   = f"s3://{args.bucket}/{args.prefix}/train"
    val_s3     = f"s3://{args.bucket}/{args.prefix}/validation"
    output_s3  = f"s3://{args.bucket}/{args.prefix}/model"

    # Get the SageMaker-managed XGBoost container image URI
    xgboost_image = sagemaker.image_uris.retrieve(
        framework='xgboost',
        region=region,
        version=XGBOOST_VERSION
    )

    # Configure estimator
    estimator = Estimator(
        image_uri=xgboost_image,
        role=role,
        instance_count=1,
        instance_type=args.instance_type,
        volume_size=30,
        output_path=output_s3,
        hyperparameters=HYPERPARAMETERS,
        sagemaker_session=sess,
    )

    # Configure data channels — label must be the first column in CSV
    train_input = TrainingInput(train_s3,   content_type='text/csv')
    val_input   = TrainingInput(val_s3,     content_type='text/csv')

    print("\nLaunching training job...")
    estimator.fit({'train': train_input, 'validation': val_input}, wait=True, logs='All')

    # Retrieve outputs
    training_job_name  = estimator.latest_training_job.name
    model_artifact_uri = estimator.model_data

    print(f"\n  Training job complete: {training_job_name}")
    print(f"  Model artifact: {model_artifact_uri}")

    # Write outputs for downstream steps
    outputs = {
        'training_job_name':  training_job_name,
        'model_artifact_uri': model_artifact_uri,
        'xgboost_version':    XGBOOST_VERSION,
        'instance_type':      args.instance_type,
        'hyperparameters':    HYPERPARAMETERS,
        'completed_at':       datetime.utcnow().isoformat(),
    }

    with open(args.output_file, 'w') as f:
        json.dump(outputs, f, indent=2)

    print(f"\nCHECKPOINT 2 PASSED — outputs written to {args.output_file}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
