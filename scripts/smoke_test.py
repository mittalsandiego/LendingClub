"""
scripts/smoke_test.py
Checkpoint 7 — Smoke Test

Sends N known test records to the live endpoint and asserts:
  (a) Each response row contains exactly 3 probability values
  (b) Probabilities sum to 1.0 within tolerance (±0.01)
  (c) p99 latency is below the threshold (default 500ms)

Exits with code 1 on any assertion failure, triggering
the rollback step in the GitHub Actions workflow.

Usage:
    python scripts/smoke_test.py \
        --endpoint-name loan-status-xgb-prod \
        --bucket sagemaker-us-east-1-099405935674 \
        --prefix aai540-group9-loan-project \
        --num-records 10 \
        --latency-threshold-ms 500 \
        --output-file smoke_test_results.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import boto3
import numpy as np
import pandas as pd


CLASS_NAMES = ['Fully Paid', 'Charged Off', 'Current']
PROB_SUM_TOLERANCE = 0.01   # Probabilities must sum to 1.0 ± this value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint-name', required=True)
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--num-records', type=int, default=10)
    parser.add_argument('--latency-threshold-ms', type=float, default=500.0)
    parser.add_argument('--output-file', default='smoke_test_results.json')
    return parser.parse_args()


def load_test_sample(bucket, prefix, n):
    """Download a small sample from the test CSV in S3."""
    s3 = boto3.client('s3')
    test_key = f"{prefix}/test/test.csv"

    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        tmp = f.name
    s3.download_file(bucket, test_key, tmp)

    df = pd.read_csv(tmp, header=None, nrows=n)
    os.unlink(tmp)

    # Column 0 is the label; columns 1+ are features
    y = df.iloc[:, 0].values.astype(int)
    X = df.iloc[:, 1:].values.astype(float)
    return X, y


def invoke_endpoint_row(runtime_client, endpoint_name, row):
    """Invoke the endpoint with a single CSV row and measure latency."""
    payload = ','.join(map(str, row))

    start = time.perf_counter()
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='text/csv',
        Body=payload
    )
    latency_ms = (time.perf_counter() - start) * 1000

    body = response['Body'].read().decode('utf-8').strip()
    probs = [float(p) for p in body.split(',')]
    return probs, latency_ms


def assert_probabilities_valid(probs, row_idx):
    """Assert a probability vector has 3 values summing to 1."""
    # Assertion (a): exactly 3 values
    if len(probs) != 3:
        raise AssertionError(
            f"Row {row_idx}: expected 3 probabilities, got {len(probs)}: {probs}"
        )

    # Assertion (b): sum to 1.0 ± tolerance
    total = sum(probs)
    if abs(total - 1.0) > PROB_SUM_TOLERANCE:
        raise AssertionError(
            f"Row {row_idx}: probabilities sum to {total:.6f}, "
            f"expected 1.0 ± {PROB_SUM_TOLERANCE}: {probs}"
        )


def main():
    args = parse_args()
    region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')

    runtime = boto3.client('sagemaker-runtime', region_name=region)

    print("\n" + "="*60)
    print("CHECKPOINT 7: SMOKE TEST")
    print("="*60)
    print(f"  Endpoint:          {args.endpoint_name}")
    print(f"  Records:           {args.num_records}")
    print(f"  Latency threshold: {args.latency_threshold_ms}ms")

    # Load test samples from S3
    print(f"\n[1/2] Loading {args.num_records} test records from S3...")
    X, y = load_test_sample(args.bucket, args.prefix, args.num_records)
    print(f"  Loaded {len(y)} records")

    # Run assertions
    print(f"\n[2/2] Invoking endpoint and running assertions...")

    latencies = []
    results   = []
    all_passed = True

    for i, row in enumerate(X):
        try:
            probs, latency_ms = invoke_endpoint_row(runtime, args.endpoint_name, row)

            # Assertion (a) + (b)
            assert_probabilities_valid(probs, i)

            predicted = CLASS_NAMES[int(np.argmax(probs))]
            actual    = CLASS_NAMES[y[i]]
            latencies.append(latency_ms)

            print(
                f"  Row {i:>2}: probs=[{probs[0]:.3f}, {probs[1]:.3f}, {probs[2]:.3f}]  "
                f"pred={predicted:<12}  actual={actual:<12}  "
                f"{latency_ms:.0f}ms  ✅"
            )
            results.append({
                'row': i, 'probabilities': probs,
                'predicted': predicted, 'actual': actual,
                'latency_ms': latency_ms, 'passed': True
            })

        except AssertionError as e:
            print(f"  Row {i:>2}:  ASSERTION FAILED: {e}")
            all_passed = False
            results.append({'row': i, 'error': str(e), 'passed': False})

        except Exception as e:
            print(f"  Row {i:>2}:  INVOCATION ERROR: {e}")
            all_passed = False
            results.append({'row': i, 'error': str(e), 'passed': False})

    # Assertion (c): latency
    if latencies:
        p99 = float(np.percentile(latencies, 99))
        p50 = float(np.percentile(latencies, 50))
        print(f"\n  Latency — p50: {p50:.0f}ms | p99: {p99:.0f}ms (threshold: {args.latency_threshold_ms}ms)")

        if p99 > args.latency_threshold_ms:
            print(f"   p99 latency {p99:.0f}ms exceeds threshold {args.latency_threshold_ms}ms")
            all_passed = False
        else:
            print(f"  Latency within threshold")
    else:
        p99, p50 = None, None

    # Write results
    smoke_results = {
        'timestamp':           datetime.utcnow().isoformat(),
        'endpoint_name':       args.endpoint_name,
        'num_records_tested':  len(results),
        'all_assertions_passed': all_passed,
        'latency_p50_ms':      p50,
        'latency_p99_ms':      p99,
        'latency_threshold_ms': args.latency_threshold_ms,
        'row_results':         results,
    }

    with open(args.output_file, 'w') as f:
        json.dump(smoke_results, f, indent=2)

    print(f"\n  Results written to {args.output_file}")
    print("\n" + "="*60)

    if all_passed:
        print("CHECKPOINT 7 PASSED — All smoke test assertions cleared")
        print("   Pipeline complete. Endpoint is serving the new model version.")
        print("="*60 + "\n")
    else:
        print(" CHECKPOINT 7 FAILED — Rolling back endpoint to previous version")
        print("="*60 + "\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
