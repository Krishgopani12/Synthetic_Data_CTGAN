import pandas as pd
import numpy as np
from ctgan import CTGAN
import os

# Define default paths
DATA_DIR = "data"
MODEL_DIR = "models"
OUTPUT_DIR = "output"

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def detect_categorical_columns(df, threshold=10):
    """
    Automatically detect categorical columns in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        threshold (int): Maximum number of unique values to consider a column categorical
        
    Returns:
        list: List of categorical column names
    """
    categorical_columns = []
    
    for column in df.columns:
        # Consider a column categorical if it's object type or has few unique values
        if df[column].dtype == 'object' or df[column].nunique() < threshold:
            categorical_columns.append(column)
            
    return categorical_columns

def evaluate_synthetic_data(real_data, synthetic_data, categorical_features):
    """
    Evaluate synthetic data quality using statistical metrics
    
    Args:
        real_data (pd.DataFrame): Original data
        synthetic_data (pd.DataFrame): Generated synthetic data
        categorical_features (list): List of categorical column names
        
    Returns:
        dict: Dictionary containing evaluation metrics in percentages
    """
    metrics = {}
    
    # Create copies of the data to avoid modifying originals
    real_data_encoded = real_data.copy()
    synthetic_data_encoded = synthetic_data.copy()
    
    # Encode categorical columns for correlation calculation
    for column in categorical_features:
        if real_data[column].dtype == 'object':
            # Get all unique values from both datasets
            all_categories = set(real_data[column].unique()) | set(synthetic_data[column].unique())
            category_map = {cat: idx for idx, cat in enumerate(all_categories)}
            
            # Encode both datasets
            real_data_encoded[column] = real_data[column].map(category_map)
            synthetic_data_encoded[column] = synthetic_data[column].map(category_map)
    
    # Column correlation similarity
    try:
        real_corr = real_data_encoded.corr()
        synt_corr = synthetic_data_encoded.corr()
        correlation_similarity = 100 * (1 - np.mean(np.abs(real_corr - synt_corr)))
        metrics['correlation_similarity'] = round(correlation_similarity, 2)
    except Exception as e:
        print(f"Warning: Could not calculate correlation similarity: {str(e)}")
        metrics['correlation_similarity'] = None
    
    # Column distribution similarity
    distribution_similarities = []
    for column in real_data.columns:
        try:
            if column in categorical_features:
                # For categorical columns, compare value distributions
                real_dist = real_data[column].value_counts(normalize=True)
                synt_dist = synthetic_data[column].value_counts(normalize=True)
                
                # Align distributions and fill missing categories with 0
                real_dist, synt_dist = real_dist.align(synt_dist, fill_value=0)
                
                similarity = 100 * (1 - np.mean(np.abs(real_dist - synt_dist)))
                distribution_similarities.append(similarity)
            else:
                # For numerical columns, compare basic statistics
                real_stats = real_data[column].describe()
                synt_stats = synthetic_data[column].describe()
                
                stats_similarity = 100 * (1 - np.mean(np.abs(
                    (real_stats - synt_stats) / real_stats
                )))
                distribution_similarities.append(stats_similarity)
        except Exception as e:
            print(f"Warning: Could not calculate distribution similarity for column {column}: {str(e)}")
            continue
    
    if distribution_similarities:
        metrics['avg_distribution_similarity'] = round(np.mean(distribution_similarities), 2)
    else:
        metrics['avg_distribution_similarity'] = None
    
    # Unique values preservation
    unique_similarities = []
    for column in real_data.columns:
        try:
            real_unique = real_data[column].nunique()
            synt_unique = synthetic_data[column].nunique()
            similarity = 100 * min(real_unique, synt_unique) / max(real_unique, synt_unique)
            unique_similarities.append(similarity)
        except Exception as e:
            print(f"Warning: Could not calculate unique value similarity for column {column}: {str(e)}")
            continue
    
    if unique_similarities:
        metrics['unique_values_preservation'] = round(np.mean(unique_similarities), 2)
    else:
        metrics['unique_values_preservation'] = None
    
    return metrics

def generate_synthetic_data(
    input_file: str = f"{DATA_DIR}/input_data.csv",
    epochs: int = 200,
    output_file: str = f"{OUTPUT_DIR}/synthetic_data.csv",
    evaluate: bool = True,
    categorical_threshold: int = 10,
    model_output: str = f"{MODEL_DIR}/ctgan_model.pkl",
    num_samples: int = 1000
):
    """
    Generate synthetic data using CTGAN for any input CSV file
    
    Args:
        input_file (str): Path to input CSV file
        epochs (int): Number of training epochs
        output_file (str): Path to save synthetic data
        evaluate (bool): Whether to evaluate the synthetic data
        categorical_threshold (int): Maximum unique values to consider a column categorical
        model_output (str): Path to save the trained model
        num_samples (int): Number of synthetic samples to generate
    """
    try:
        # Read the original data
        print(f"Reading data from {input_file}")
        data = pd.read_csv(input_file)
        
        # Automatically detect categorical features
        categorical_features = detect_categorical_columns(data, categorical_threshold)
        print(f"\nDetected categorical columns: {categorical_features}")
        
        # Handle missing values
        for column in data.columns:
            if data[column].dtype == 'object':
                data[column] = data[column].fillna('missing')
            else:
                data[column] = data[column].fillna(data[column].mean())
        
        # Initialize and train CTGAN
        print(f"\nTraining CTGAN for {epochs} epochs...")
        ctgan = CTGAN(verbose=True)
        ctgan.fit(data, categorical_features, epochs=epochs)
        
        # Save the trained model if path is provided
        if model_output:
            print(f"\nSaving trained model to {model_output}")
            ctgan.save(model_output)
        
        # Generate synthetic samples
        print(f"\nGenerating {num_samples} synthetic samples...")
        synthetic_data = ctgan.sample(num_samples)
        
        # Post-process synthetic data to match original data types
        for column in data.columns:
            if data[column].dtype in ['int64', 'int32']:
                synthetic_data[column] = synthetic_data[column].round().astype(data[column].dtype)
        
        # Save synthetic data if output path is provided
        if output_file:
            synthetic_data.to_csv(output_file, index=False)
            print(f"Synthetic data saved to {output_file}")
        
        # Evaluate the synthetic data if requested
        if evaluate:
            print("\nEvaluating synthetic data quality...")
            metrics = evaluate_synthetic_data(data, synthetic_data, categorical_features)
            print("\nEvaluation Metrics (in percentages):")
            for metric_name, value in metrics.items():
                if value is not None:
                    print(f"{metric_name.replace('_', ' ').title()}: {value}%")
                else:
                    print(f"{metric_name.replace('_', ' ').title()}: Could not calculate")
        
        return synthetic_data
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def generate_from_saved_model(
    model_path: str = f"{MODEL_DIR}/ctgan_model.pkl",
    num_samples: int = 1000,
    output_file: str = f"{OUTPUT_DIR}/synthetic_data_from_model.csv"
):
    """
    Generate synthetic data using a saved CTGAN model
    
    Args:
        model_path (str): Path to the saved CTGAN model
        num_samples (int): Number of synthetic samples to generate
        output_file (str): Path to save synthetic data
    """
    try:
        # Load the saved model
        print(f"Loading model from {model_path}")
        ctgan = CTGAN.load(model_path)
        
        # Generate synthetic samples
        print(f"\nGenerating {num_samples} synthetic samples...")
        synthetic_data = ctgan.sample(num_samples)
        
        # Save synthetic data if output path is provided
        if output_file:
            synthetic_data.to_csv(output_file, index=False)
            print(f"Synthetic data saved to {output_file}")
            
        return synthetic_data
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def main():
    """
    Main function to either train a new model or generate data from saved model
    """
    # Get number of samples from user first
    while True:
        try:
            num_samples = int(input("\nEnter the number of synthetic samples to generate: "))
            if num_samples > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")

    # Ask user what they want to do
    print("\nWhat would you like to do?")
    print("1. Train new model and generate data")
    print("2. Load existing model and generate data")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1 or 2): "))
            if choice in [1, 2]:
                break
            else:
                print("Please enter either 1 or 2.")
        except ValueError:
            print("Please enter a valid number.")
    
    if choice == 1:
        # Train new model and generate data
        synthetic_data = generate_synthetic_data(num_samples=num_samples)
    else:
        synthetic_data = generate_from_saved_model(num_samples=num_samples)

if __name__ == "__main__":
    main()