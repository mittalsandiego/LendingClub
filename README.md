# Loan Status Prediction ML System 

> **AAI-540 Machine Learning Operations | Group 9 | University of San Diego**  
> Multi-class loan outcome classification on AWS SageMaker using LendWise Analytics LendingClub 2016 data

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Business Context](#2-business-context)
3. [Architecture](#3-architecture)
4. [Dataset](#4-dataset)
5. [Repository Structure](#5-repository-structure)
6. [AWS Infrastructure & Resources](#6-aws-infrastructure--resources)
7. [Environment Setup](#7-environment-setup)
8. [Notebook Walkthrough](#8-notebook-walkthrough)
   - [Section 1: Environment Setup & Data Ingestion](#section-1-environment-setup--data-ingestion)
   - [Section 2: Exploratory Data Analysis](#section-2-exploratory-data-analysis)
   - [Section 3: Athena SQL Data Lake Queries](#section-3-athena-sql-data-lake-queries)
   - [Section 4: Feature Engineering & Feature Store](#section-4-feature-engineering--feature-store)
   - [Section 5: Model Training & Evaluation](#section-5-model-training--evaluation)
   - [Section 6: Model Registry & Model Card](#section-6-model-registry--model-card)
   - [Section 7: Model Deployment with Data Capture](#section-7-model-deployment-with-data-capture)
   - [Section 8: Model Quality Monitoring](#section-8-model-quality-monitoring)
   - [Section 9: Cleanup](#section-9-cleanup)
9. [Actual Run Results](#9-actual-run-results)
10. [Known Limitations & Design Decisions](#10-known-limitations--design-decisions)
11. [Future Enhancements](#11-future-enhancements)
12. [MLOps Component Checklist](#12-mlops-component-checklist)
13. [Team](#13-team)
14. [References](#14-references)

---

## 1. Project Overview

This repository contains the complete codebase for **LendWise Analytics**, a production-style MLOps pipeline built on AWS SageMaker. The system trains a multi-class XGBoost classifier to predict the outcome of individual LendingClub loans into one of three categories:

| Label | Class | Description |
|-------|-------|-------------|
| `0` | **Fully Paid** | Borrower repaid the loan in full |
| `1` | **Charged Off** | Borrower defaulted; lender wrote off the loss |
| `2` | **Current** | Loan is active and payments are being made |

**The primary objective of this project is to demonstrate a complete, production-ready MLOps pipeline** — covering every stage from raw data ingestion to automated model monitoring — rather than to optimize predictive accuracy on an inherently imbalanced dataset. This is consistent with the AAI-540 course focus on ML engineering practices over ML performance benchmarking.

---

## 2. Business Context

LendWise Analytics is a hypothetical fintech company helping credit investors and lending institutions make faster, more consistent loan risk decisions at scale. Manual credit review cannot keep pace with high application volumes. By automating loan outcome prediction, LendWise Analytics enables:

- **Loan officers** to focus review time on borderline cases flagged by the model
- **Risk analysts** to monitor portfolio-level trends in predicted charge-off rates
- **Compliance teams** to audit model behavior through captured inference logs

The business scenario deliberately mirrors real-world credit risk systems, which means it carries real-world regulatory implications under FCRA, ECOA, and CCPA — documented in the companion ethics analysis (Discussion 7.1).

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                         │
│   Kaggle CSV (4 files) ──► Amazon S3 ──► AWS Glue Catalog       │
│                                  └──► Amazon Athena (SQL)       │
└─────────────────────────────────────┬───────────────────────────┘
                                      │
┌─────────────────────────────────────▼───────────────────────────┐
│                   FEATURE ENGINEERING LAYER                     │
│   Pandas Preprocessing ──► SageMaker Feature Store             │
│        ├── Loan Feature Group    (21 features, 50k records)     │
│        └── Borrower Feature Group (14 features, 50k records)    │
└─────────────────────────────────────┬───────────────────────────┘
                                      │
┌─────────────────────────────────────▼───────────────────────────┐
│                     MODEL TRAINING LAYER                        │
│   S3 (train/val/test CSVs) ──► SageMaker Training Job           │
│        └── XGBoost 1.7-1, ml.m5.xlarge, multi:softprob          │
└─────────────────────────────────────┬───────────────────────────┘
                                      │
┌─────────────────────────────────────▼───────────────────────────┐
│                   MODEL GOVERNANCE LAYER                        │
│   model.tar.gz ──► SageMaker Model Registry                     │
│        ├── Model Package Group: loan-status-prediction-xgboost  │
│        ├── Version 1 (PendingManualApproval)                     │
│        └── Model Card: loan-status-xgboost-model-card           │
└─────────────────────────────────────┬───────────────────────────┘
                                      │
┌─────────────────────────────────────▼───────────────────────────┐
│                    DEPLOYMENT LAYER                             │
│   Approved Model ──► SageMaker Real-Time Endpoint               │
│        └── Data Capture (100%) ──► S3 (inference logs)          │
└─────────────────────────────────────┬───────────────────────────┘
                                      │
┌─────────────────────────────────────▼───────────────────────────┐
│                    MONITORING LAYER                             │
│   S3 Logs ──► SageMaker Model Quality Monitor                   │
│        ├── Baseline: validation set (7,500 records)             │
│        ├── Schedule: hourly cron                                │
│        └── CloudWatch Alarm: accuracy < 0.75                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Dataset

### Source
- **Kaggle Dataset:** [sarahvch/predicting-who-pays-back-loans](https://www.kaggle.com/datasets/sarahvch/predicting-who-pays-back-loans)
- **Original Source:** LendingClub quarterly loan statistics, 2016 Q1–Q4
- **Files Used:**
  - `LoanStats_2016Q1.csv`
  - `LoanStats_2016Q2.csv`
  - `LoanStats_2016Q3.csv`
  - `LoanStats_2016Q4.csv`

### Dataset Statistics

| Property | Value |
|----------|-------|
| Raw records (4 files combined) | 434,415 rows |
| Columns | 111 |
| After class filtering | 421,185 rows |
| Training sample (Feature Store) | 50,000 rows (stratified) |
| Training split | 35,000 rows (70%) |
| Validation split | 7,500 rows (15%) |
| Test split | 7,500 rows (15%) |

### Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Current | 377,897 | 89.72% |
| Fully Paid | 36,368 | 8.63% |
| Charged Off | 6,920 | 1.64% |

> **Note on Class Imbalance:** The 2016 dataset is dominated by `Current` loans because most loans from that year were still actively being repaid at the time of data collection. This creates a severe class imbalance that the model reflects in its predictions. See [Section 10](#10-known-limitations--design-decisions) for full discussion.

### Key Columns Used

| Column | Type | Description |
|--------|------|-------------|
| `loan_amnt` | float | Total loan amount requested |
| `funded_amnt` | float | Actual funded amount |
| `term` | int | Loan term (36 or 60 months) |
| `int_rate` | float | Interest rate (%) |
| `installment` | float | Monthly payment |
| `grade` | string | LendingClub loan grade (A–G) |
| `purpose` | string | Loan purpose (debt_consolidation, etc.) |
| `annual_inc` | float | Borrower annual income |
| `emp_length` | string | Years of employment |
| `dti` | float | Debt-to-income ratio |
| `fico_range_low/high` | float | FICO credit score bounds |
| `delinq_2yrs` | float | Delinquencies in past 2 years |
| `revol_util` | float | Revolving credit utilization % |
| `home_ownership` | string | RENT / OWN / MORTGAGE |
| `loan_status` | string | **Target variable** |

---

## 5. Repository Structure

```
aai540-group9-loan-project/
│
├── LoanStatus_MLOps_Project.ipynb    # Main notebook — complete MLOps pipeline
├── README.md                         # This file
│
├── data/                             # Data files (stored in S3; local copies optional)
│   ├── LoanStats_2016Q1.csv          # Raw quarterly data files (download from Kaggle)
│   ├── LoanStats_2016Q2.csv
│   ├── LoanStats_2016Q3.csv
│   └── LoanStats_2016Q4.csv
│
└── docs/
    ├── LoanStatus_ML_Design_Document_v2.docx   # ML System Design Document
    └── Discussion_7_1_Ethics_Group9.docx       # Ethics & privacy law analysis
```

> **Data Note:** Raw CSV files are not committed to GitHub due to size. Download them from Kaggle ([sarahvch/predicting-who-pays-back-loans](https://www.kaggle.com/datasets/sarahvch/predicting-who-pays-back-loans)) and place them in the project root directory or set the `LOAN_DATA_DIR` environment variable to their location before running the notebook.

---

## 6. AWS Infrastructure & Resources

All AWS resources created during the notebook run are listed below for reference and reproducibility.

### IAM & Region
| Resource | Value |
|----------|-------|
| IAM Role | `arn:aws:iam::099405935674:role/LabRole` |
| Region | `us-east-1` |
| Account | `099405935674` |

### S3 Paths
| Purpose | S3 Path |
|---------|---------|
| S3 Bucket | `sagemaker-us-east-1-099405935674` |
| Project prefix | `aai540-group9-loan-project/` |
| Raw data | `.../data/loan_data_raw.csv` |
| Parquet data lake | `.../data/` (Parquet partitions) |
| Athena results staging | `.../athena-results/` |
| Feature Store — Loan | `.../feature-store/loan/` |
| Feature Store — Borrower | `.../feature-store/borrower/` |
| Training data | `.../train/train.csv` |
| Validation data | `.../validation/validation.csv` |
| Test data | `.../test/test.csv` |
| Model artifact | `.../model/model.tar.gz` |
| Data Capture | `.../data-capture/` |
| Monitoring baseline data | `.../monitoring/baseline/data/` |
| Monitoring baseline results | `.../monitoring/baseline/results/` |

### Glue / Athena
| Resource | Value |
|----------|-------|
| Glue Database | `aai540_vmittalATsandiego_loan` |
| Glue Table | `loan_data` |

### SageMaker Feature Store
| Feature Group | Name | Records Ingested |
|---------------|------|-----------------|
| Loan features | `loan-features-21-12-11-57` | 50,000 |
| Borrower features | `borrower-features-21-12-11-57` | 50,000 |

### SageMaker Training
| Resource | Value |
|----------|-------|
| Training Job | `sagemaker-xgboost-2026-02-21-12-22-16-379` |
| Framework | XGBoost 1.7-1 |
| Instance | `ml.m5.xlarge` |

### SageMaker Model Registry
| Resource | Value |
|----------|-------|
| Model Package Group | `loan-status-prediction-xgboost` |
| Group ARN | `arn:aws:sagemaker:us-east-1:099405935674:model-package-group/loan-status-prediction-xgboost` |
| Version 1 ARN | `arn:aws:sagemaker:us-east-1:099405935674:model-package/loan-status-prediction-xgboost/1` |
| Approval Status | `PendingManualApproval` |
| Model Card | `loan-status-xgboost-model-card` |
| Model Card ARN | `arn:aws:sagemaker:us-east-1:099405935674:model-card/loan-status-xgboost-model-card` |

### SageMaker Endpoint
| Resource | Value |
|----------|-------|
| Endpoint Name | `loan-status-xgb-2026-02-21-1254` |
| Instance | `ml.m5.large` |
| Data Capture | Enabled — 100% sampling |

### SageMaker Model Monitor
| Resource | Value |
|----------|-------|
| Baseline Job | `loan-quality-baseline-2026-02-21-1259` |
| Monitor Schedule | `loan-quality-monitor-2026-02-21-1308` |
| Schedule ARN | `arn:aws:sagemaker:us-east-1:099405935674:monitoring-schedule/loan-quality-monitor-2026-02-21-1308` |
| Monitor Type | `ModelQuality` |
| Schedule | Hourly |
| Status | Pending (activates at next hour boundary) |

### CloudWatch
| Resource | Value |
|----------|-------|
| Alarm Name | `LOAN_STATUS_MODEL_QUALITY_ACCURACY` |
| Metric | `accuracy` |
| Threshold | `< 0.75` |
| Namespace | `aws/sagemaker/Endpoints/model-metrics` |

---

## 7. Environment Setup

### Prerequisites

- AWS Account with SageMaker access
- SageMaker Studio or SageMaker Notebook Instance (recommended: `ml.t3.medium` or larger)
- IAM role with permissions for: SageMaker, S3, Glue, Athena, CloudWatch, Feature Store
- Python 3.9+ kernel

### Install Dependencies

Run the first cell in the notebook, or manually install:

```bash
pip install awswrangler pyathena "sagemaker==2.*" seaborn "xgboost==2.*" --quiet
```

### Required Python Libraries

| Library | Purpose |
|---------|---------|
| `boto3` | AWS SDK — S3, SageMaker, CloudWatch API calls |
| `sagemaker` | SageMaker Python SDK — training, deployment, monitoring |
| `pandas` | Data loading, manipulation, splitting |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | Visualization — EDA charts, confusion matrix |
| `awswrangler` | S3 ↔ Parquet conversion, Glue Catalog registration |
| `pyathena` | Athena SQL queries from Python |
| `xgboost` | Local model training and evaluation |
| `sklearn` | train_test_split, classification_report, confusion_matrix |

### Dataset Setup

1. Download the dataset from Kaggle:
   ```
   https://www.kaggle.com/datasets/sarahvch/predicting-who-pays-back-loans
   ```
2. Place all four CSV files in the notebook directory:
   ```
   LoanStats_2016Q1.csv
   LoanStats_2016Q2.csv
   LoanStats_2016Q3.csv
   LoanStats_2016Q4.csv
   ```
3. Alternatively, set the `LOAN_DATA_DIR` environment variable to the directory containing the files:
   ```bash
   export LOAN_DATA_DIR=/path/to/data/
   ```

---

## 8. Notebook Walkthrough

The single notebook `LoanStatus_MLOps_Project.ipynb` is organized into 9 sequential sections. Each section builds on the previous one and must be run in order.

---

### Section 1: Environment Setup & Data Ingestion

**Purpose:** Configure the AWS environment, load the raw data, upload to S3, and register it in the Glue Data Catalog for Athena querying.

**What it does:**
1. Initializes a `sagemaker.Session()` and retrieves the default S3 bucket, IAM role, and region
2. Defines all S3 path variables using project prefix `aai540-group9-loan-project`
3. Loads all four LoanStats CSV files, concatenates them, and cleans percentage string columns (`int_rate`, `revol_util`) to float
4. Filters the combined dataset to the three target classes: `Fully Paid`, `Charged Off`, `Current`
5. Uploads the filtered CSV to S3
6. Uses `awswrangler.s3.to_parquet()` to convert to Parquet and register the Glue table `loan_data` in database `aai540_vmittalATsandiego_loan`

**Key outputs:**
```
Combined LoanStats shape: (434415, 111)
Filtered dataset shape: (421185, 111)
Raw data uploaded to: s3://sagemaker-us-east-1-099405935674/aai540-group9-loan-project/data/loan_data_raw.csv
Database "aai540_vmittalATsandiego_loan" ready.
Table "loan_data" registered in Glue Catalog.
```

**Why Parquet?** Columnar format with compression — 3–5x faster for Athena analytical queries vs CSV, and significantly cheaper since Athena charges per bytes scanned.

---

### Section 2: Exploratory Data Analysis

**Purpose:** Understand data distributions, class balance, missing values, and feature relationships before modeling.

**Charts produced:**
1. **Class Distribution Bar + Pie Chart** — visualizes the 89.7% / 8.6% / 1.6% class split
2. **Missing Value Bar Chart** — identifies 8 columns with >50% missing values (dropped in preprocessing)
3. **Feature Distribution Histograms** — `loan_amnt`, `int_rate`, `annual_inc`, `dti` overlaid by class
4. **Charge-Off Rate by Loan Grade** — bar chart showing grade A (0.44%) to grade G (10.60%) default progression

**Key EDA findings:**

| Finding | Implication |
|---------|-------------|
| 89.7% of loans are `Current` | Severe class imbalance — model will predict majority class |
| Grade G loans charge off at 10.6% vs Grade A at 0.44% | `grade_encoded` is the strongest predictive feature |
| Charged Off loans have avg interest rate 16.67% vs 12.85% for Current | `int_rate` is a strong class separator |
| 8 columns have >50% missing values | These are dropped before feature engineering |
| Debt consolidation = 57% of all loan purposes | `purpose` needs one-hot encoding, not ordinal |

---

### Section 3: Athena SQL Data Lake Queries

**Purpose:** Demonstrate the data lake capability — querying raw data directly from S3 via Athena SQL without loading into memory.

**5 queries executed:**

| Query | Business Question | Key Result |
|-------|-------------------|------------|
| Q1 | Class distribution | Current 89.72%, Fully Paid 8.63%, Charged Off 1.64% |
| Q2 | Avg metrics by class | Charged Off has highest avg rate (16.67%) and DTI (21.59%) |
| Q3 | Charge-off rate by grade | Grade A: 0.44% → Grade G: 10.60% — clear monotonic relationship |
| Q4 | Top 10 loan purposes | Debt consolidation dominates (240,909 loans); small_business has highest risk |
| Q5 | Charge-off by income band | Low income (<$40k): 2.07% vs Very High (>$150k): 1.25% |

These queries demonstrate that even without building a model, the data lake surfaces actionable credit risk insights — validating the business value of the ingestion pipeline.

---

### Section 4: Feature Engineering & SageMaker Feature Store

**Purpose:** Engineer features from raw columns and persist them in SageMaker Feature Store across two Feature Groups that mirror real organizational data domains.

#### Feature Engineering Steps

**Loan Feature Group transformations:**
- `term`: `"36 months"` → `36.0` (regex extract)
- `grade`: ordinal encoded → `grade_encoded` (`A=1, B=2, ..., G=7`)
- `purpose`: one-hot encoded → 14 binary columns (`purpose_debt_consolidation`, `purpose_credit_card`, etc.)
- `loan_status`: label encoded → `label` (`Fully Paid=0, Charged Off=1, Current=2`)
- Added: `loan_id_str` (string record identifier), `event_time` (Unix timestamp — required by Feature Store)

**Borrower Feature Group transformations:**
- `emp_length`: `"10+ years"` → `10.0`, `"< 1 year"` → `0.0` (regex extract)
- `fico_range_low` + `fico_range_high` → averaged into `fico_avg`
- `home_ownership`: one-hot encoded → `home_MORTGAGE`, `home_OWN`, `home_RENT`
- Added: `borrower_id_str`, `event_time`

**Dropped features:**
- 8 columns with >50% missing values
- Leakage-risk columns (`total_rec_prncp`, `recoveries`, `last_pymnt_amnt`)
- Free-text (`emp_title`, `title`, `desc`)
- ID/URL fields (`id`, `member_id`, `url`)

#### Feature Store Creation

```python
# Feature Groups created:
# loan-features-21-12-11-57     → 21 features, 50,000 records ingested
# borrower-features-21-12-11-57 → 14 features, 50,000 records ingested
```

Both Feature Groups are created with:
- **Online Store enabled** — low-latency record lookups during real-time inference
- **Offline Store in S3** — Parquet files for batch training retrieval

**Verification — sample Feature Store record retrieved:**
```
Feature Store record for Loan ID: 83345159
  loan_amnt: 10000.0
  int_rate: 12.79
  grade_encoded: 3
  purpose_debt_consolidation: 1
  label: 2
  ...
```

---

### Section 5: Model Training & Evaluation

**Purpose:** Train an XGBoost multi-class classifier using the SageMaker managed training infrastructure.

#### Data Split
| Split | Records | Use |
|-------|---------|-----|
| Train | 35,000 | Model fitting |
| Validation | 7,500 | Early stopping |
| Test | 7,500 | Final evaluation |

Data is uploaded to S3 as CSV with **label as the first column** (required by SageMaker XGBoost container format).

#### Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| `objective` | `multi:softprob` | Outputs probability per class |
| `num_class` | `3` | Three loan outcome classes |
| `num_round` | `150` | Max boosting rounds |
| `max_depth` | `6` | Balance complexity vs overfitting |
| `eta` | `0.1` | Conservative learning rate |
| `subsample` | `0.8` | Row sampling per tree |
| `colsample_bytree` | `0.8` | Feature sampling per tree |
| `eval_metric` | `merror` | Multi-class error rate |
| `early_stopping_rounds` | `10` | Stop if val does not improve |

#### Training Job
```
Training Job: sagemaker-xgboost-2026-02-21-12-22-16-379
Instance: ml.m5.xlarge
Started: 2026-02-21 12:22:18 UTC
Duration: ~2.5 minutes
```

#### Evaluation Results (Test Set, 7,500 records)

```
=== Classification Report ===
              precision    recall  f1-score   support

  Fully Paid       0.00      0.00      0.00       650
 Charged Off       0.00      0.00      0.00       117
     Current       0.90      1.00      0.95      6733

    accuracy                           0.90      7500
   macro avg       0.30      0.33      0.32      7500
weighted avg       0.81      0.90      0.85      7500

Overall Test Accuracy: 0.8976
```

#### Top 5 Feature Importances
```
1. grade_encoded   (highest)
2. int_rate
3. term
4. installment
5. funded_amnt
```

> See [Section 10](#10-known-limitations--design-decisions) for the full explanation of why accuracy is 90% but minority-class F1 is 0.

---

### Section 6: Model Registry & Model Card

**Purpose:** Register the trained model with version control and document it formally in SageMaker Model Registry.

#### Part 1 — Model Package Group

```python
# Group: loan-status-prediction-xgboost
# ARN: arn:aws:sagemaker:us-east-1:099405935674:model-package-group/loan-status-prediction-xgboost
# Created: 2026-02-21 12:30:31 UTC
```

#### Part 2 — Versioned Model Package

```python
# Version: 1
# ARN: arn:aws:sagemaker:us-east-1:099405935674:model-package/loan-status-prediction-xgboost/1
# Status: PendingManualApproval
# Container: SageMaker XGBoost 1.7-1
# Artifact: s3://.../aai540-group9-loan-project/model/model.tar.gz
# Input: text/csv | Output: text/csv (probabilities)
# Supported instances: ml.m5.large, ml.m5.xlarge
```

The `PendingManualApproval` status simulates a human review gate — in a production CI/CD pipeline, a team member would inspect the Model Card and approve or reject before deployment proceeds.

#### Part 3 — Model Card

```python
# Model Card: loan-status-xgboost-model-card
# ARN: arn:aws:sagemaker:us-east-1:099405935674:model-card/loan-status-xgboost-model-card
# Status: Draft
# Version: 1
```

The Model Card documents:
- Model description and creator
- Intended uses and risk rating (Medium)
- Business problem and stakeholders
- Training details and hyperparameters
- Evaluation metrics (accuracy: 0.8976)
- Ethical considerations (proxy variables, class imbalance liability)
- Caveats and recommendations

---

### Section 7: Model Deployment with Data Capture

**Purpose:** Deploy the model as a real-time SageMaker endpoint with 100% Data Capture to feed the monitoring pipeline.

```python
# Endpoint: loan-status-xgb-2026-02-21-1254
# Instance: ml.m5.large
# Data Capture: 100% to s3://.../data-capture/
```

#### Inference Format

**Input:** CSV row, no header, features in training order  
**Output:** Three comma-separated probabilities `[P(Fully Paid), P(Charged Off), P(Current)]`

**Sample response for 5 test rows:**
```
0.157, 0.026, 0.817  →  Current
0.086, 0.004, 0.910  →  Current
0.027, 0.002, 0.971  →  Current
0.081, 0.003, 0.916  →  Current
0.079, 0.005, 0.916  →  Current
```

The prediction is the class with the highest probability (`argmax`).

---

### Section 8: Model Quality Monitoring

**Purpose:** Configure continuous automated monitoring that compares live predictions against a validated baseline and alerts on drift.

#### Baseline Creation

A baseline dataset of 7,500 validation predictions with ground truth labels is uploaded to S3 and processed by a SageMaker baselining job:

```
Baseline Job: loan-quality-baseline-2026-02-21-1259
Baseline Shape: 7,500 rows × 3 columns (probability, prediction, label)
Completed: 2026-02-21 13:03:24 UTC
```

**Baseline statistics (from SageMaker output):**
```json
{
  "multiclass_classification_metrics": {
    "accuracy": { "value": 0.8976, "standard_deviation": 0.0008481 },
    "confusion_matrix": {
      "2→2": 6732,  "0→2": 650,  "1→2": 115,
      "2→0": 2,     "0→0": 0,    "1→0": 1
    }
  }
}
```

#### Monitoring Schedule

```python
# Schedule: loan-quality-monitor-2026-02-21-1308
# ARN: arn:aws:sagemaker:us-east-1:099405935674:monitoring-schedule/loan-quality-monitor-2026-02-21-1308
# Type: ModelQuality
# Cron: Hourly
# Status: Pending
```

#### CloudWatch Alarm

```python
# Alarm: LOAN_STATUS_MODEL_QUALITY_ACCURACY
# Metric: accuracy
# Namespace: aws/sagemaker/Endpoints/model-metrics
# Threshold: < 0.75
# Period: 1 hour
# Treatment of missing: notBreaching
```

When triggered, this alarm indicates either data drift (input feature distributions have shifted) or concept drift (the relationship between features and outcomes has changed), both of which require investigation and potential retraining.

---

### Section 9: Cleanup

The cleanup section contains commented-out cells to delete AWS resources when the project is complete:

```python
# quality_monitor.delete_monitoring_schedule()
# predictor.delete_model()
# predictor.delete_endpoint()
```

> **Do not run cleanup until after the video demonstration is recorded.** The live endpoint and monitoring schedule are required for the video deliverable.

---

## 9. Actual Run Results

All results below are from the actual notebook execution on 2026-02-21.

| Component | Result |
|-----------|--------|
| Raw dataset loaded | 434,415 rows × 111 columns |
| After class filter | 421,185 rows |
| S3 upload | Done Successful |
| Glue/Athena registration | Done Database and table created |
| Athena queries (5) | Done All returned results |
| Loan Feature Group | Done Created + 50,000 records ingested |
| Borrower Feature Group | Done Created + 50,000 records ingested |
| Feature Store read-back | Done Loan ID 83345159 retrieved correctly |
| SageMaker Training Job | Done Completed in ~2.5 min |
| Test Accuracy | 89.76% |
| Fully Paid F1 | 0.00 (class imbalance — see Section 10) |
| Charged Off F1 | 0.00 (class imbalance — see Section 10) |
| Current F1 | 0.95 |
| Top feature | `grade_encoded` |
| Model Package Group | Done Created |
| Model Package Version 1 | Done Registered (PendingManualApproval) |
| Model Card | Done Created (Draft, Version 1) |
| Endpoint deployed | Done `loan-status-xgb-2026-02-21-1254` |
| Data Capture | Done Enabled (100%) |
| Baseline job | Done Completed |
| Monitoring schedule | Done Created (Pending/Hourly) |
| CloudWatch alarm | Done Created (threshold: accuracy < 0.75) |

---

## 10. Known Limitations & Design Decisions

### Class Imbalance (Primary Limitation)

The 2016 LendingClub dataset has a 90% / 8.6% / 1.6% class split because most loans from that year were still active ("Current") when the data was collected. The model learns to predict `Current` for virtually all inputs, achieving 90% accuracy while being completely unable to identify `Fully Paid` or `Charged Off` loans (F1 = 0.00 for both minority classes).

**This is a known and intentional tradeoff in this project** because:

1. The course objective is MLOps pipeline demonstration, not ML accuracy optimization
2. The full pipeline (Feature Store, Model Registry, Data Capture, Monitoring) works correctly regardless of model quality
3. The class imbalance is honestly documented in the Model Card, the Design Document, and this README — which itself is good MLOps practice
4. The monitoring system will correctly detect when this behavior changes

**Legal note:** As analyzed in Discussion 7.1, a model that never predicts `Charged Off` has ECOA implications — it would never deny a loan on risk grounds, which could itself constitute discriminatory lending by approving all applicants regardless of risk. This reinforces why the imbalance must be addressed before any commercial use.

### 50,000 Record Sample for Feature Store

The full 421,185-record dataset is not ingested into Feature Store — only a stratified 50,000-record sample. This is a practical decision to reduce ingestion time and SageMaker Feature Store costs in an academic environment. The full dataset is available in S3 and Athena for analytical queries.

### Single-Instance Endpoint

The deployed endpoint runs on a single `ml.m5.large` instance. For production workloads, auto-scaling would be configured. This is intentionally simple for the academic demonstration.

---

## 11. Future Enhancements

| Priority | Enhancement | Expected Impact |
|----------|-------------|-----------------|
| 1 | **Class imbalance remediation** — SMOTE oversampling, class-weighted XGBoost, or switch to 2007–2011 dataset where loans have fully matured | Dramatically improve Fully Paid and Charged Off F1 scores |
| 2 | **SageMaker Clarify integration** — disparate impact analysis across income bands for ECOA compliance | Required before any commercial deployment |
| 3 | **Hyperparameter optimization (SageMaker AMT)** — Bayesian search over `max_depth`, `eta`, `subsample` | Incremental accuracy improvement |
| 4 | **Data Quality Monitor** — detect input feature drift (e.g., avg `int_rate` shift) as leading indicator of model degradation | Earlier drift detection before accuracy drops |
| 5 | **Full-dataset Feature Store ingestion** — scale from 50k to 421k records with incremental quarterly updates | Production-representative Feature Store |
| 6 | **Feature enrichment with macroeconomic data** — add unemployment rate, Fed funds rate per Rajaraman's principle that more data beats better algorithms | Improved predictive power without algorithm changes |
| 7 | **CI/CD automation via GitHub Actions** — automate the full pipeline from data validation through deployment with human approval gates | Reproducible, auditable retraining workflow |

---

## 12. MLOps Component Checklist

This project was designed to demonstrate every major MLOps component covered in AAI-540. The table below maps each component to its implementation.

| MLOps Component | Implementation | Status |
|-----------------|---------------|--------|
| Data Ingestion | S3 upload + Parquet conversion | Done |
| Data Lake | AWS Glue Catalog + Amazon Athena | Done |
| SQL Analytics | 5 Athena queries on S3 data | Done |
| Feature Engineering | Pandas transforms (encoding, cleaning) | Done |
| Feature Store | SageMaker Feature Store, 2 groups | Done |
| Model Training | SageMaker Managed Training Job | Done |
| Experiment Tracking | SageMaker training job logs + metrics | Done |
| Model Evaluation | sklearn classification_report + confusion matrix | Done |
| Model Artifact Storage | model.tar.gz in S3 | Done |
| Model Registry | SageMaker Model Package Group + versioned package | Done |
| Model Governance | SageMaker Model Card (Draft) | Done |
| Human Approval Gate | PendingManualApproval status in registry | Done |
| Model Deployment | SageMaker Real-Time Endpoint | Done |
| Inference Logging | SageMaker Data Capture (100%) | Done |
| Model Monitoring | SageMaker Model Quality Monitor | Done |
| Monitoring Baseline | Validation set baseline job | Done |
| Automated Alerting | CloudWatch Alarm on accuracy metric | Done |
| Infrastructure Metrics | CloudWatch endpoint metrics (auto) | Done |
| Data Ethics Analysis | Discussion 7.1 — FCRA, ECOA, CCPA, ADPPA | Done |
| CI/CD Pipeline | Designed in Design Document; not automated | Planned |
| Bias Detection | SageMaker Clarify integration | Planned |
| Data Quality Monitor | Feature drift detection | Planned |

---

## 13. Team

**Group 9 — AAI-540 Machine Learning Operations**  
University of San Diego — Applied Artificial Intelligence

| Member | Email |
|--------|-------|
| Vinay Mittal | [vmittal@sandiego.edu |

---

## 14. References

- Amazon Web Services. (2024). *Amazon SageMaker Feature Store documentation*. https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html

- Amazon Web Services. (2024). *Amazon SageMaker Model Monitor documentation*. https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html

- Amazon Web Services. (2024). *Amazon SageMaker Model Registry documentation*. https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html

- Amazon Web Services. (2024). *SageMaker XGBoost built-in algorithm*. https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html

- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785

- Consumer Financial Protection Bureau. (2022). *CFPB circular 2022-03: Adverse action notification requirements and the Fair Credit Reporting Act*. https://www.consumerfinance.gov/compliance/circulars/circular-2022-03/

- Rajaraman, A. (2008, March 24). More data usually beats better algorithms. *Datawocky*. https://anand.typepad.com/datawocky/2008/03/more-data-usual.html

- sarahvch. (2024). *Predicting who pays back loans* [Dataset]. Kaggle. https://www.kaggle.com/datasets/sarahvch/predicting-who-pays-back-loans

---

*Thank you*
