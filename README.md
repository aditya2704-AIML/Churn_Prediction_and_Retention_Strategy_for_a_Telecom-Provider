# Churn_Prediction_and_Retention_Strategy_for_a_Telecom-Provider
An end-to-end machine learning pipeline leveraging SMOTE, hyperparameter optimization, and a Stacking Ensemble to predict telecom customer churn and drive proactive retention strategies.

## Project Overview
Customer churn is a critical challenge in the telecom industry, directly impacting revenue, customer lifetime value, and market share. This project presents an end-to-end data science workflow designed to predict customer churn, identify the underlying business drivers, and provide actionable, segment-level retention strategies.

The pipeline iterates from foundational exploratory data analysis and baseline models to advanced, hyperparameter-tuned ensembles and deep learning architectures, prioritizing the most cost-sensitive business metric: **Recall**.

## Table of Contents
1. [Dataset Details](#dataset-details)
2. [Tech Stack](#tech-stack)
3. [Project Workflow](#project-workflow)
4. [Key Findings & Business Impact](#key-findings--business-impact)
5. [The Winning Model](#the-winning-model)
6. [How to Run](#how-to-run)

## Dataset Details
* **Source File:** `telecom_churn.csv` (25,000 records, 36 features)
* **Target Variable:** `is_churn` (1 = Churned, 0 = Retained)
* **Feature Categories:** Demographics, Usage and Value Behavior, Billing and Payment, Service Experience, and Digital Engagement.

## Tech Stack
* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn`, `xgboost`, `imbalanced-learn` (SMOTE)
* **Deep Learning:** `tensorflow`, `keras`

## Project Workflow

### Phase 1: Data Processing & Baseline Modeling
* **Data Validation:** Handled missing values, removed duplicates, and evaluated class imbalances.
* **Exploratory Data Analysis (EDA):** Analyzed churn rates across different customer segments (e.g., tenure, plan type, contract structure).
* **Feature Engineering:** Created complex business-logic features, including:
  * *Overage Ratio* and *Spend per Tenure* (Financial Stress Indicators)
  * *Network Dissatisfaction* (Service Quality Interactions)
  * *Complaint Escalation* (Customer Frustration Trends)
* **Baseline Models:** Built and evaluated initial Logistic Regression and Random Forest models.

### Phase 2: Advanced Modeling & Optimization
* **Strategic Threshold Tuning:** Shifted the classification threshold (e.g., to 40%) to drastically improve recall and capture a higher volume of true churners.
* **Handling Class Imbalance:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to perfectly balance the training data.
* **Hyperparameter Tuning:** Utilized `GridSearchCV` to find the optimal architecture for an `XGBClassifier`.
* **Stacking Ensemble:** Built a robust meta-model combining the strengths of Random Forest, XGBoost, and Logistic Regression.
* **Deep Learning Benchmark:** Designed and trained a multi-layer Neural Network (with Batch Normalization and Dropout) using TensorFlow/Keras to test the absolute limits of predictive power and capture deep non-linearities.

## Key Findings & Business Impact
* **Primary Churn Drivers:** Feature importance extraction revealed that Net Promoter Score (NPS), Total Charges, Average Data Speed, and Overage Charges heavily influence churn behavior.
* **Segment Risks:** Newer customers and those on month-to-month or prepaid plans exhibit significantly higher churn risk.
* **Retention Strategy:** The model outputs allow the marketing team to deploy highly targeted campaigns:
  * *High-value at-risk:* Proactive retention outreach.
  * *Payment stress:* Billing support and flexible payment plans.
  * *Service issues:* Improved network resolution and complaint handling.

## The Winning Model
The **Stacking Ensemble** was selected as the final deployment-ready model. While the Deep Learning architecture provided a high-ceiling benchmark, the tree-based Stacking Ensemble offered the ideal balance of predictive accuracy (maximizing recall) and **business interpretability**. By extracting clear feature importances, the ensemble allows the business to understand not just *who* will churn, but *why*.
