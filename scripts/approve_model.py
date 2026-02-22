"""
scripts/approve_model.py
Checkpoint 5 helper — Approves a Model Package in SageMaker Model Registry.
Called after the human approval gate clears in GitHub Actions.

Usage:
    python scripts/approve_model.py \
        --model-package-arn arn:aws:sagemaker:... \
        --approval-comment "Approved via CI/CD — commit abc1234"
"""

import argparse
import os
from datetime import datetime

import boto3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-package-arn', required=True)
    parser.add_argument('--approval-comment', default='Approved via automated CI/CD pipeline')
    return parser.parse_args()


def main():
    args = parse_args()
    region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
    sm_client = boto3.client('sagemaker', region_name=region)

    print(f"\nApproving Model Package: {args.model_package_arn}")

    sm_client.update_model_package(
        ModelPackageArn=args.model_package_arn,
        ModelApprovalStatus='Approved',
        ApprovalDescription=args.approval_comment
    )

    print(f"Model Package approved at {datetime.utcnow().isoformat()}")
    print(f"   Comment: {args.approval_comment}")


if __name__ == '__main__':
    main()
