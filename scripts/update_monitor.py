"""
scripts/update_monitor.py
Checkpoint 6 helper — Updates the Model Quality Monitor baseline
and schedule after a new model version is deployed.

Usage:
    python scripts/update_monitor.py \
        --endpoint-name loan-status-xgb-prod \
        --bucket sagemaker-us-east-1-099405935674 \
        --prefix aai540-group9-loan-project \
        --accuracy-threshold 0.75
"""

import argparse
import os
from datetime import datetime

import boto3
import sagemaker
from sagemaker.model_monitor import ModelQualityMonitor, CronExpressionGenerator, EndpointInput
from sagemaker.model_monitor.dataset_format import DatasetFormat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint-name', required=True)
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--accuracy-threshold', type=float, default=0.75)
    return parser.parse_args()


def main():
    args = parse_args()
    region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
    role   = os.environ.get('SAGEMAKER_ROLE', sagemaker.get_execution_role())
    sess   = sagemaker.Session()

    run_ts = datetime.utcnow().strftime('%Y-%m-%d-%H%M')

    baseline_data_uri    = f's3://{args.bucket}/{args.prefix}/monitoring/baseline/data'
    baseline_results_uri = f's3://{args.bucket}/{args.prefix}/monitoring/baseline/results'

    print(f"\nUpdating Model Quality Monitor for endpoint: {args.endpoint_name}")

    quality_monitor = ModelQualityMonitor(
        role=role,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        volume_size_in_gb=20,
        max_runtime_in_seconds=1800,
        sagemaker_session=sess
    )

    # Re-run baseline against the new model
    baseline_job_name = f'loan-quality-baseline-{run_ts}'
    print(f"  Running baseline job: {baseline_job_name}...")

    quality_monitor.suggest_baseline(
        job_name=baseline_job_name,
        dataset_uri=f'{baseline_data_uri}/baseline_with_predictions.csv',
        dataset_format=DatasetFormat.csv(header=True),
        output_s3_uri=baseline_results_uri,
        problem_type='MulticlassClassification',
        inference_attribute='prediction',
        ground_truth_attribute='label',
        wait=True,
        logs=False
    )
    print("  Baseline job complete")

    # Create a new monitoring schedule pointing to the updated baseline
    schedule_name = f'loan-quality-monitor-{run_ts}'
    ep_input = EndpointInput(
        endpoint_name=args.endpoint_name,
        destination='/opt/ml/processing/input_data',
        inference_attribute='0',
        probability_attribute='0',
        probability_threshold_attribute=0.5
    )

    quality_monitor.create_monitoring_schedule(
        monitor_schedule_name=schedule_name,
        endpoint_input=ep_input,
        output_s3_uri=f's3://{args.bucket}/{args.prefix}/monitoring/results',
        problem_type='MulticlassClassification',
        ground_truth_input=f's3://{args.bucket}/{args.prefix}/monitoring/ground-truth',
        constraints=quality_monitor.suggested_constraints(),
        schedule_cron_expression=CronExpressionGenerator.hourly(),
        enable_cloudwatch_metrics=True,
    )
    print(f"  Monitoring schedule created: {schedule_name}")
    print(f"     Accuracy alert threshold: < {args.accuracy_threshold}")


if __name__ == '__main__':
    main()
