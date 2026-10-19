# Practice 4: Synthetic Data Generation with Faker and Stratified Train-Test Split

## Description

This project generates a synthetic customer dataset using Faker, pandas, and NumPy. It prepares the data for machine learning by encoding categorical variables and splitting the dataset into training and testing sets using stratified sampling.

## Features

- Generation of 1,000 synthetic customer records.
- Stratified train-test split with an 80/20 ratio.
- Comparison of churn class distributions across the full dataset, training set, and test set.
- Export of generated data and results to CSV files.
- Random seeds for reproducible data generation and partitioning.

## Requirements

- Python 3.8+
- Faker
- pandas
- NumPy
- scikit-learn

## Installation

1. Clone this repository and navigate to the project directory.
2. Create a Python virtual environment:

   ```bash
   python3 -m venv venv
   ```

3. Activate the virtual environment:

   ```bash
   source venv/bin/activate
   ```

4. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Execution

Run the script from the project root directory:

```bash
python scripts/synthetic_data.py
```

The program pauses after each step. Press Enter to continue through random seed configuration, synthetic data generation, feature and target separation, categorical encoding, stratified splitting, class distribution comparison, and file export.

## Generated Files

The script creates the following CSV files in the project root directory:

- `synthetic_customers.csv`: The complete synthetic customer dataset.
- `train_data.csv`: The training features and target.
- `test_data.csv`: The testing features and target.
- `churn_distribution.csv`: The churn class proportions for the full dataset, training set, and test set.
