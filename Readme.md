# Synthetic Data Generator using CTGAN

This project implements a synthetic data generation system using CTGAN (Conditional Tabular GAN), which can create realistic synthetic data while preserving statistical properties of the original dataset.

## Dataset

This project uses the PaySim Financial Dataset:
- **Source**: [PaySim synthetic financial dataset on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)
- **Description**: PaySim simulates mobile money transactions based on a sample of real transactions extracted from one month of financial logs from a mobile money service implemented in an African country.
- **Features**: The dataset includes information about financial transactions including:
  - Type of transaction
  - Amount
  - Customer information
  - Merchant information
  - Fraud labels

To use this dataset:
1. Download the dataset from Kaggle
2. Place the CSV file in the `data/` directory
3. Rename it to `input_data.csv` or update the input file path in the script

Note: You'll need a Kaggle account to download the dataset.

## Features

- Automatic detection of categorical columns
- Customizable number of training epochs and samples
- Model saving and loading capabilities
- Comprehensive evaluation metrics for synthetic data quality
- Handles both numerical and categorical data
- Automatic handling of missing values

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── data/               # Directory for input data
├── models/            # Directory for saved models
├── output/           # Directory for generated synthetic data
├── generate_synthetic_data.py
├── requirements.txt
└── README.md
```

## Usage

### Running the Script

Run the script using:
```bash
python generate_synthetic_data.py
```

The script will prompt you to:
1. Enter the number of synthetic samples to generate
2. Choose between:
   - Training a new model and generating data
   - Loading an existing model and generating data

### Input Data Format

- Place your input CSV file in the `data/` directory
- Default input file name: `input_data.csv`
- The script automatically handles:
  - Categorical column detection
  - Missing value imputation
  - Data type preservation

### Output

The script generates:
- Synthetic data saved to `output/synthetic_data.csv`
- Trained model saved to `models/ctgan_model.pkl`
- Evaluation metrics including:
  - Correlation similarity
  - Distribution similarity
  - Unique values preservation

## Evaluation Metrics

The synthetic data is evaluated using three main metrics:

1. **Correlation Similarity**: Measures how well the relationships between variables are preserved
2. **Distribution Similarity**: Compares the statistical distributions of real and synthetic data
3. **Unique Values Preservation**: Evaluates how well the synthetic data maintains the variety of values

All metrics are reported as percentages, where higher values indicate better quality.

## Configuration

Key parameters that can be modified in the script:

- `epochs`: Number of training epochs (default: 200)
- `categorical_threshold`: Maximum unique values to consider a column categorical (default: 10)
- `num_samples`: Number of synthetic samples to generate

## Dependencies

- pandas
- numpy
- ctgan
- sdv
