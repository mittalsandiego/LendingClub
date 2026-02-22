"""
scripts/register_model.py
Checkpoint 4 — Model Registration

Registers the evaluated model in SageMaker Model Registry as a
versioned Model Package under the 'loan-status-prediction-xgboost'
Model Package Group. Status is set to PendingManualApproval to
enforce the human review gate (Checkpoint 5).

Also updates the SageMaker Model Card with current run metrics,
git SHA, and evaluation results.

Usage:
    python scripts/register_model.py \
        --bucket sagemaker-us-east-1-099405935674 \
        --model-artifact-uri s3://.../model.tar.gz \
        --model-package-group loan-status-prediction-xgboost \
        --training-job-name sagemaker-xgboost-2026-... \
        --evaluation-report evaluation_report.json \
        --git-sha abc1234 \
        --output-file registration_outputs.json
"""

import argparse
import json
import os
from datetime import datetime

import boto3
import sagemaker


MODEL_CARD_NAME = 'loan-status-xgboost-model-card'

# SageMaker XGBoost 1.7-1 container — matches the training container
XGBOOST_VERSION = '1.7-1'


def parse_args():
    parser = argparse.ArgumentParser(description='Register model in SageMaker Model Registry')
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--model-artifact-uri', required=True)
    parser.add_argument('--model-package-group', required=True)
    parser.add_argument('--training-job-name', required=True)
    parser.add_argument('--evaluation-report', required=True)
    parser.add_argument('--git-sha', default='unknown')
    parser.add_argument('--output-file', default='registration_outputs.json')
    return parser.parse_args()


def get_xgboost_image_uri(region):
    """Retrieve the SageMaker-managed XGBoost container image URI."""
    return sagemaker.image_uris.retrieve(
        framework='xgboost',
        region=region,
        version=XGBOOST_VERSION
    )


def ensure_model_package_group_exists(sm_client, group_name):
    """Create the Model Package Group if it doesn't already exist."""
    try:
        sm_client.describe_model_package_group(ModelPackageGroupName=group_name)
        print(f"  Model Package Group '{group_name}' already exists.")
    except sm_client.exceptions.ResourceNotFound:
        sm_client.create_model_package_group(
            ModelPackageGroupName=group_name,
            ModelPackageGroupDescription=(
                'XGBoost multi-class model for LendingClub loan status prediction. '
                'Classes: Fully Paid (0), Charged Off (1), Current (2). '
                'AAI-540 Group 9 — LendWise Analytics.'
            )
        )
        print(f"  Model Package Group '{group_name}' created.")


def register_model_package(sm_client, args, eval_report, image_uri, region):
    """Register a versioned Model Package with PendingManualApproval."""
    accuracy    = eval_report.get('accuracy', 0)
    weighted_f1 = eval_report.get('weighted_f1', 0)
    timestamp   = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')

    response = sm_client.create_model_package(
        ModelPackageGroupName=args.model_package_group,
        ModelPackageDescription=(
            f'XGBoost v1 — multi-class loan status classifier. '
            f'Accuracy: {accuracy:.4f}, Weighted F1: {weighted_f1:.4f}. '
            f'Trained: {timestamp}. Git SHA: {args.git_sha}.'
        ),
        ModelApprovalStatus='PendingManualApproval',
        InferenceSpecification={
            'Containers': [{
                'Image': image_uri,
                'ModelDataUrl': args.model_artifact_uri,
                'Framework': 'XGBOOST',
                'FrameworkVersion': '1.7',
                'NearestModelName': 'xgboost',
            }],
            'SupportedTransformInstanceTypes': ['ml.m5.large', 'ml.m5.xlarge'],
            'SupportedRealtimeInferenceInstanceTypes': ['ml.m5.large', 'ml.m5.xlarge'],
            'SupportedContentTypes': ['text/csv'],
            'SupportedResponseMIMETypes': ['text/csv'],
        },
        ModelMetrics={
            'ModelQuality': {
                'Statistics': {
                    'ContentType': 'application/json',
                    'S3Uri': (
                        f's3://{args.bucket}/aai540-group9-loan-project/'
                        f'monitoring/baseline/results/statistics.json'
                    )
                }
            }
        },
        CustomerMetadataProperties={
            'accuracy':           str(round(accuracy, 4)),
            'weighted_f1':        str(round(weighted_f1, 4)),
            'training_job':       args.training_job_name,
            'git_sha':            args.git_sha,
            'registered_at':      timestamp,
            'framework':          f'XGBoost {XGBOOST_VERSION}',
            'target_classes':     'Fully Paid, Charged Off, Current',
        }
    )

    return response['ModelPackageArn']


def update_model_card(sm_client, eval_report, model_package_arn, git_sha):
    """Update the existing Model Card with the latest run metrics."""
    accuracy    = eval_report.get('accuracy', 0)
    weighted_f1 = eval_report.get('weighted_f1', 0)
    timestamp   = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')

    card_content = {
        'model_overview': {
            'model_description': (
                'XGBoost multi-class classifier predicting LendingClub loan outcomes '
                '(Fully Paid, Charged Off, Current) from borrower and loan features.'
            ),
            'model_creator': 'LendWise Analytics — AAI-540 Group 9',
            'problem_type': 'MulticlassClassification',
        },
        'intended_uses': {
            'purpose_of_model': 'Loan underwriting risk support — flag high-risk applications.',
            'intended_users': ['Loan officers', 'Risk analysts', 'Portfolio managers'],
            'use_guidelines': (
                'Must not be the sole decision-maker for loan approvals. '
                'Adverse action notices required per FCRA. '
                'Disparate impact audit required before commercial deployment (ECOA).'
            ),
            'risk_rating': 'Medium',
        },
        'training_details': {
            'objective_function': 'multi:softprob',
            'training_observations': eval_report.get('test_set_size', 'N/A'),
            'training_job_name': git_sha,
        },
        'evaluation_details': [{
            'name': f'CI/CD Run — {timestamp}',
            'evaluation_observation': (
                f'Accuracy: {accuracy:.4f} | Weighted F1: {weighted_f1:.4f} | '
                f'Git SHA: {git_sha}'
            ),
            'metric_groups': [{
                'name': 'Classification Metrics',
                'metric_data': [
                    {'name': 'accuracy',    'type': 'number', 'value': accuracy},
                    {'name': 'weighted_f1', 'type': 'number', 'value': weighted_f1},
                ]
            }]
        }],
        'ethical_considerations': {
            'risk': (
                'Features annual_inc, dti, and revol_util may proxy protected attributes '
                '(race, national origin) under ECOA. Loan grade may encode historical '
                'lending bias. Class imbalance (89.7% Current) means model rarely predicts '
                'minority classes — this is a documented limitation requiring remediation '
                'before commercial use. See Discussion 7.1 ethics analysis.'
            ),
            'mitigations': (
                'SageMaker Clarify disparate impact analysis planned. '
                'Model Card and Design Document disclose class imbalance. '
                'Human review required for all adverse decisions.'
            ),
        },
        'additional_information': {
            'caveats_and_recommendations': (
                f'Model Package ARN: {model_package_arn}. '
                f'Deployed at: {timestamp}. '
                f'Class imbalance (Priority 1 future enhancement): apply SMOTE or '
                f'switch to 2007-2011 dataset for balanced Fully Paid / Charged Off split.'
            )
        }
    }

    try:
        # Try to update existing card first
        sm_client.update_model_card(
            ModelCardName=MODEL_CARD_NAME,
            Content=json.dumps(card_content)
        )
        print(f"  Model Card '{MODEL_CARD_NAME}' updated.")
    except sm_client.exceptions.ResourceNotFound:
        # Create if not found
        sm_client.create_model_card(
            ModelCardName=MODEL_CARD_NAME,
            Content=json.dumps(card_content),
            ModelCardStatus='Draft'
        )
        print(f"  Model Card '{MODEL_CARD_NAME}' created.")


def main():
    args = parse_args()
    region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')

    sm_client = boto3.client('sagemaker', region_name=region)
    image_uri = get_xgboost_image_uri(region)

    with open(args.evaluation_report) as f:
        eval_report = json.load(f)

    print("\n" + "="*60)
    print("CHECKPOINT 4: MODEL REGISTRATION")
    print("="*60)
    print(f"  Model Package Group: {args.model_package_group}")
    print(f"  Model Artifact:      {args.model_artifact_uri}")
    print(f"  Git SHA:             {args.git_sha}")

    # Ensure the package group exists
    print("\n[1/3] Ensuring Model Package Group exists...")
    ensure_model_package_group_exists(sm_client, args.model_package_group)

    # Register versioned model package
    print("\n[2/3] Registering Model Package (PendingManualApproval)...")
    model_package_arn = register_model_package(
        sm_client, args, eval_report, image_uri, region
    )

    # Extract version number from ARN (last segment)
    version = model_package_arn.split('/')[-1]
    print(f"  Registered: {model_package_arn}")
    print(f"  Version: {version}")
    print(f"  Status: PendingManualApproval")

    # Update Model Card
    print("\n[3/3] Updating Model Card...")
    update_model_card(sm_client, eval_report, model_package_arn, args.git_sha)

    # Write outputs
    outputs = {
        'model_package_arn':     model_package_arn,
        'model_package_version': version,
        'model_package_group':   args.model_package_group,
        'approval_status':       'PendingManualApproval',
        'git_sha':               args.git_sha,
        'registered_at':         datetime.utcnow().isoformat(),
    }

    with open(args.output_file, 'w') as f:
        json.dump(outputs, f, indent=2)

    print(f"\n  Registration outputs written to {args.output_file}")
    print("\nCHECKPOINT 4 PASSED — Model registered, awaiting human approval")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
