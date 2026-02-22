"""
scripts/deploy_model.py
Checkpoint 6 — Endpoint Deployment

Deploys an approved Model Package to a SageMaker real-time endpoint
using an atomic endpoint config swap (zero-downtime blue/green update).

Strategy:
  1. Create a new EndpointConfig with the approved model
  2. If endpoint exists: call update_endpoint() — SageMaker handles
     blue/green traffic routing automatically
  3. If endpoint does not exist: call create_endpoint()
  4. Wait for InService status
  5. Store the old config name for rollback
  6. Enable Data Capture at 100% sampling (matches design doc)

Usage:
    python scripts/deploy_model.py \
        --model-package-arn arn:aws:sagemaker:... \
        --endpoint-name loan-status-xgb-prod \
        --instance-type ml.m5.large \
        --bucket sagemaker-us-east-1-099405935674 \
        --prefix aai540-group9-loan-project \
        --output-file deployment_outputs.json
"""

import argparse
import json
import os
import time
from datetime import datetime

import boto3
import sagemaker


DATA_CAPTURE_SAMPLING_PCT = 100   # 100% capture — matches design document Section 7


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-package-arn', required=True)
    parser.add_argument('--endpoint-name', required=True)
    parser.add_argument('--instance-type', default='ml.m5.large')
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--output-file', default='deployment_outputs.json')
    return parser.parse_args()


def create_model_from_package(sm_client, model_package_arn, role, run_id):
    """Create a SageMaker Model object from the approved Model Package."""
    model_name = f'lendwise-xgb-{run_id}'

    sm_client.create_model(
        ModelName=model_name,
        Containers=[{'ModelPackageName': model_package_arn}],
        ExecutionRoleArn=role,
    )
    print(f"  SageMaker Model created: {model_name}")
    return model_name


def create_endpoint_config(sm_client, model_name, instance_type, bucket, prefix, run_id):
    """Create an EndpointConfig with Data Capture enabled."""
    config_name = f'lendwise-xgb-config-{run_id}'
    capture_s3_uri = f's3://{bucket}/{prefix}/data-capture'

    sm_client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            'VariantName': 'primary',
            'ModelName': model_name,
            'InstanceType': instance_type,
            'InitialInstanceCount': 1,
            'InitialVariantWeight': 1.0,
        }],
        DataCaptureConfig={
            'EnableCapture': True,
            'InitialSamplingPercentage': DATA_CAPTURE_SAMPLING_PCT,
            'DestinationS3Uri': capture_s3_uri,
            'CaptureOptions': [
                {'CaptureMode': 'Input'},
                {'CaptureMode': 'Output'},
            ],
            'CaptureContentTypeHeader': {
                'CsvContentTypes': ['text/csv'],
            }
        }
    )
    print(f"  EndpointConfig created: {config_name}")
    print(f"     Data Capture: {DATA_CAPTURE_SAMPLING_PCT}% → {capture_s3_uri}")
    return config_name


def endpoint_exists(sm_client, endpoint_name):
    """Check if the endpoint already exists."""
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        return True
    except sm_client.exceptions.ClientError:
        return False


def get_current_endpoint_config(sm_client, endpoint_name):
    """Retrieve the current endpoint config name for rollback storage."""
    try:
        desc = sm_client.describe_endpoint(EndpointName=endpoint_name)
        return desc['EndpointConfigName']
    except Exception:
        return None


def wait_for_endpoint(sm_client, endpoint_name, timeout_seconds=600):
    """Poll until the endpoint reaches InService or Failed status."""
    print(f"\n  Waiting for endpoint '{endpoint_name}' to reach InService status...")
    elapsed = 0
    poll_interval = 20

    while elapsed < timeout_seconds:
        desc = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = desc['EndpointStatus']
        print(f"  [{elapsed:>3}s] Status: {status}")

        if status == 'InService':
            return True
        if status in ('Failed', 'OutOfService'):
            reason = desc.get('FailureReason', 'Unknown')
            print(f"   Endpoint reached terminal state: {status}. Reason: {reason}")
            return False

        time.sleep(poll_interval)
        elapsed += poll_interval

    print(f"   Timeout after {timeout_seconds}s waiting for endpoint")
    return False


def main():
    args = parse_args()
    region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
    role   = os.environ.get('SAGEMAKER_ROLE', sagemaker.get_execution_role())

    sm_client = boto3.client('sagemaker', region_name=region)
    run_id    = datetime.utcnow().strftime('%Y%m%d-%H%M%S')

    print("\n" + "="*60)
    print("CHECKPOINT 6: ENDPOINT DEPLOYMENT")
    print("="*60)
    print(f"  Endpoint:      {args.endpoint_name}")
    print(f"  Model Package: {args.model_package_arn}")
    print(f"  Instance:      {args.instance_type}")

    # Save old config name for rollback
    old_config = get_current_endpoint_config(sm_client, args.endpoint_name)
    if old_config:
        print(f"  Previous config (rollback target): {old_config}")

    # Create new SageMaker Model from approved package
    print("\n[1/3] Creating SageMaker Model from approved package...")
    model_name = create_model_from_package(sm_client, args.model_package_arn, role, run_id)

    # Create new EndpointConfig with Data Capture
    print("\n[2/3] Creating new EndpointConfig with Data Capture...")
    new_config = create_endpoint_config(
        sm_client, model_name, args.instance_type, args.bucket, args.prefix, run_id
    )

    # Deploy or update endpoint
    print("\n[3/3] Deploying to endpoint...")
    if endpoint_exists(sm_client, args.endpoint_name):
        print(f"  Endpoint exists — performing zero-downtime update...")
        sm_client.update_endpoint(
            EndpointName=args.endpoint_name,
            EndpointConfigName=new_config,
        )
    else:
        print(f"  Endpoint does not exist — creating new endpoint...")
        sm_client.create_endpoint(
            EndpointName=args.endpoint_name,
            EndpointConfigName=new_config,
        )

    # Wait for InService
    success = wait_for_endpoint(sm_client, args.endpoint_name)

    if not success:
        print("\n CHECKPOINT 6 FAILED — endpoint did not reach InService")
        print(f"   Rollback target: {old_config}")
        # Write rollback info to file for the rollback script
        with open('rollback_config.json', 'w') as f:
            json.dump({
                'endpoint_name': args.endpoint_name,
                'rollback_config': old_config
            }, f)
        raise RuntimeError("Endpoint deployment failed — triggering rollback")

    print(f"\n  Endpoint '{args.endpoint_name}' is InService")

    # Write outputs
    outputs = {
        'endpoint_name':    args.endpoint_name,
        'model_name':       model_name,
        'endpoint_config':  new_config,
        'previous_config':  old_config,
        'model_package_arn': args.model_package_arn,
        'deployed_at':      datetime.utcnow().isoformat(),
    }

    with open(args.output_file, 'w') as f:
        json.dump(outputs, f, indent=2)

    print("\nCHECKPOINT 6 PASSED — Endpoint deployed successfully")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
