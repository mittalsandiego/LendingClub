"""
scripts/rollback_endpoint.py
Rollback helper — Reverts a SageMaker endpoint to its previous
EndpointConfig when the smoke test (Checkpoint 7) fails.

Reads the previous config name from deployment_outputs.json
(written by deploy_model.py) and calls update_endpoint()
to restore the prior model version.

Usage:
    python scripts/rollback_endpoint.py \
        --endpoint-name loan-status-xgb-prod \
        --bucket sagemaker-us-east-1-099405935674 \
        --prefix aai540-group9-loan-project
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import boto3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint-name', required=True)
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--prefix', required=True)
    return parser.parse_args()


def main():
    args   = parse_args()
    region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
    sm     = boto3.client('sagemaker', region_name=region)

    print("\n" + "="*60)
    print("ROLLBACK: Reverting endpoint to previous version")
    print("="*60)

    # Read the previous config name written by deploy_model.py
    deployment_file = 'deployment_outputs.json'
    if not os.path.exists(deployment_file):
        print(f"   {deployment_file} not found — cannot determine rollback target")
        sys.exit(1)

    with open(deployment_file) as f:
        deployment = json.load(f)

    previous_config = deployment.get('previous_config')
    if not previous_config:
        print("  ⚠️  No previous config found in deployment outputs — endpoint may be brand new")
        print("     Deleting endpoint instead of rolling back...")
        sm.delete_endpoint(EndpointName=args.endpoint_name)
        print(f"  Endpoint '{args.endpoint_name}' deleted")
        return

    print(f"  Endpoint:        {args.endpoint_name}")
    print(f"  Rollback target: {previous_config}")

    # Restore previous endpoint config
    sm.update_endpoint(
        EndpointName=args.endpoint_name,
        EndpointConfigName=previous_config,
    )

    # Wait for InService
    print("  Waiting for rollback to complete...")
    elapsed = 0
    while elapsed < 600:
        desc   = sm.describe_endpoint(EndpointName=args.endpoint_name)
        status = desc['EndpointStatus']
        print(f"  [{elapsed:>3}s] Status: {status}")
        if status == 'InService':
            print(f"\n  Rollback complete — endpoint is InService with previous version")
            print(f"     Rolled back at: {datetime.utcnow().isoformat()}")
            print("="*60 + "\n")
            return
        if status in ('Failed', 'OutOfService'):
            print(f"   Rollback failed with status: {status}")
            print(f"     Failure reason: {desc.get('FailureReason', 'Unknown')}")
            sys.exit(1)
        time.sleep(20)
        elapsed += 20

    print("   Timeout waiting for rollback — manual intervention required")
    sys.exit(1)


if __name__ == '__main__':
    main()
