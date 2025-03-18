import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np

def extract_data(file_path):
    """Extract data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Handle missing values and encode categorical data."""
    
    # Handling missing values
    imputer = SimpleImputer(strategy='mean')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # Encoding categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
    
    # Dropping original categorical columns and concatenating encoded columns
    df = df.drop(columns=categorical_cols).reset_index(drop=True)
    df = pd.concat([df, encoded_df], axis=1)
    
    return df

def transform_data(df):
    """Apply feature scaling."""
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df)
    return df

def load_data(df, output_file):
    """Save the transformed data to a CSV file."""
    df.to_csv(output_file, index=False)
    print(f"Data successfully saved to {output_file}")

if __name__ == "__main__":
    input_file = "raw_data.csv"  # Replace with actual file path
    output_file = "processed_data.csv"
    
    # ETL Process
    df = extract_data(input_file)
    df = preprocess_data(df)
    df = transform_data(df)
    load_data(df, output_file)
