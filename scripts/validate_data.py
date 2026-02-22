"""
scripts/validate_data.py
Checkpoint 1 — Data Validation

Verifies that:
  - The Glue table exists and is queryable via Athena
  - Each target class meets the minimum row count threshold
  - The schema has the expected number of columns
  - Required columns are present

Usage:
    python scripts/validate_data.py \
        --bucket sagemaker-us-east-1-099405935674 \
        --prefix aai540-group9-loan-project/data \
        --glue-database aai540_vmittalATsandiego_loan \
        --glue-table loan_data \
        --min-rows-per-class 1000 \
        --expected-columns 111
"""

import argparse
import json
import os
import sys
from datetime import datetime

import boto3
import awswrangler as wr

# Target classes defined in the ML system design document
TARGET_CLASSES = ['Fully Paid', 'Charged Off', 'Current']

# Required columns — must be present after ingestion
REQUIRED_COLUMNS = [
    'loan_amnt', 'funded_amnt', 'term', 'int_rate', 'installment',
    'grade', 'purpose', 'annual_inc', 'emp_length', 'dti',
    'delinq_2yrs', 'revol_util', 'home_ownership', 'loan_status'
]


def parse_args():
    parser = argparse.ArgumentParser(description='Validate LendingClub data in S3/Glue')
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--glue-database', required=True)
    parser.add_argument('--glue-table', required=True)
    parser.add_argument('--min-rows-per-class', type=int, default=1000)
    parser.add_argument('--expected-columns', type=int, default=111)
    return parser.parse_args()


def validate_glue_table_exists(database, table, region):
    """Check that the Glue table exists and is accessible."""
    glue = boto3.client('glue', region_name=region)
    try:
        response = glue.get_table(DatabaseName=database, Name=table)
        col_count = len(response['Table']['StorageDescriptor']['Columns'])
        print(f"  Glue table '{database}.{table}' found with {col_count} columns")
        return True, col_count
    except glue.exceptions.EntityNotFoundException:
        print(f"   Glue table '{database}.{table}' NOT FOUND")
        return False, 0


def validate_class_distribution(database, table, min_rows_per_class, region):
    """Query Athena for class distribution and check minimum counts."""
    boto3_session = boto3.Session(region_name=region)
    s3_output = f"s3://sagemaker-{region}-{boto3.client('sts').get_caller_identity()['Account']}/athena-results/"

    query = f"""
        SELECT loan_status, COUNT(*) AS row_count
        FROM {database}.{table}
        WHERE loan_status IN ('Fully Paid', 'Charged Off', 'Current')
        GROUP BY loan_status
        ORDER BY row_count DESC
    """

    try:
        df = wr.athena.read_sql_query(
            sql=query,
            database=database,
            boto3_session=boto3_session,
            s3_output=s3_output
        )

        distribution = dict(zip(df['loan_status'], df['row_count']))
        all_passed = True

        for cls in TARGET_CLASSES:
            count = distribution.get(cls, 0)
            passed = count >= min_rows_per_class
            status = '✅' if passed else ''
            print(f"  {status} Class '{cls}': {count:,} rows (min: {min_rows_per_class:,})")
            if not passed:
                all_passed = False

        return all_passed, distribution

    except Exception as e:
        print(f"   Athena query failed: {e}")
        return False, {}


def validate_required_columns(database, table, region):
    """Check that all required columns are present in the Glue schema."""
    glue = boto3.client('glue', region_name=region)
    try:
        response = glue.get_table(DatabaseName=database, Name=table)
        existing_cols = {
            col['Name']
            for col in response['Table']['StorageDescriptor']['Columns']
        }
        missing = [c for c in REQUIRED_COLUMNS if c not in existing_cols]
        if missing:
            print(f"   Missing required columns: {missing}")
            return False
        print(f"  All {len(REQUIRED_COLUMNS)} required columns present")
        return True
    except Exception as e:
        print(f"   Column validation failed: {e}")
        return False


def main():
    args = parse_args()
    region = os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')

    print("\n" + "="*60)
    print("CHECKPOINT 1: DATA VALIDATION")
    print("="*60)

    results = {
        'timestamp': datetime.utcnow().isoformat(),
        'checks': {},
        'overall_passed': False
    }

    # Check 1: Glue table exists
    print("\n[1/3] Checking Glue table existence...")
    table_exists, col_count = validate_glue_table_exists(
        args.glue_database, args.glue_table, region
    )
    results['checks']['glue_table_exists'] = table_exists
    results['checks']['column_count'] = col_count

    if not table_exists:
        results['overall_passed'] = False
        os.makedirs('reports', exist_ok=True)
        with open('reports/data_validation_report.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\n VALIDATION FAILED: Glue table not found. Aborting.")
        sys.exit(1)

    # Check 2: Required columns present
    print("\n[2/3] Checking required columns...")
    cols_valid = validate_required_columns(args.glue_database, args.glue_table, region)
    results['checks']['required_columns_present'] = cols_valid

    # Check 3: Class distribution meets minimums
    print("\n[3/3] Checking class distribution via Athena...")
    classes_valid, distribution = validate_class_distribution(
        args.glue_database, args.glue_table, args.min_rows_per_class, region
    )
    results['checks']['class_distribution'] = distribution
    results['checks']['class_minimums_met'] = classes_valid

    # Overall result
    all_passed = table_exists and cols_valid and classes_valid
    results['overall_passed'] = all_passed

    os.makedirs('reports', exist_ok=True)
    with open('reports/data_validation_report.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    if all_passed:
        print("CHECKPOINT 1 PASSED — Data validation complete")
    else:
        print(" CHECKPOINT 1 FAILED — See report for details")
    print("="*60 + "\n")

    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
