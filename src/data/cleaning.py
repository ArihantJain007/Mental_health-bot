import os
import pandas as pd
import numpy as np


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Loads raw dataset from specified CSV path."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at: {filepath}")
    df = pd.read_csv(filepath)
    return df


def inspect_raw_data(df: pd.DataFrame) -> dict:
    """Computes comprehensive summary statistics for the dataset."""
    raw_rows = len(df)
    cols = df.columns.tolist()
    null_counts = df.isnull().sum().to_dict()
    
    # Exact full row duplicates (statement + status)
    if 'statement' in df.columns and 'status' in df.columns:
        full_row_dups = df.duplicated(subset=['statement', 'status']).sum()
        dup_statements = df.duplicated(subset=['statement']).sum()
        
        # Conflicting statements (same text, different status)
        valid_df = df.dropna(subset=['statement'])
        conflict_counts = valid_df.groupby('statement')['status'].nunique()
        conflicting_texts = conflict_counts[conflict_counts > 1].index.tolist()
        num_conflicting_statements = len(conflicting_texts)
        num_conflicting_rows = valid_df[valid_df['statement'].isin(conflicting_texts)].shape[0]
    else:
        full_row_dups = 0
        dup_statements = 0
        num_conflicting_statements = 0
        num_conflicting_rows = 0
        conflicting_texts = []
        
    label_dist = df['status'].value_counts(dropna=False).to_dict() if 'status' in df.columns else {}
    
    return {
        "raw_rows": raw_rows,
        "columns": cols,
        "null_counts": null_counts,
        "full_row_duplicates": int(full_row_dups),
        "duplicate_statements": int(dup_statements),
        "conflicting_statement_count": int(num_conflicting_statements),
        "conflicting_rows_count": int(num_conflicting_rows),
        "conflicting_texts": conflicting_texts,
        "label_distribution": label_dist
    }


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Executes conservative dataset cleaning:
    1. Removes index artifact 'Unnamed: 0' if present.
    2. Strips leading/trailing whitespace from statement and status strings.
    3. Removes null and empty statements.
    4. Removes all instances of statements assigned conflicting labels.
    5. Removes exact duplicate statements (keeps first occurrence).
    """
    df_clean = df.copy()
    
    # 1. Drop Unnamed: 0 if present
    if 'Unnamed: 0' in df_clean.columns:
        df_clean = df_clean.drop(columns=['Unnamed: 0'])
        
    # 2. Normalize whitespace in status and statement
    if 'status' in df_clean.columns:
        df_clean['status'] = df_clean['status'].astype(str).str.strip()
    if 'statement' in df_clean.columns:
        df_clean['statement'] = df_clean['statement'].apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )
        
    # 3. Drop missing/null/empty statements
    df_clean = df_clean.dropna(subset=['statement'])
    df_clean = df_clean[df_clean['statement'].str.len() > 0]
    
    # 4. Remove statements with conflicting labels
    conflict_counts = df_clean.groupby('statement')['status'].nunique()
    conflicting_statements = set(conflict_counts[conflict_counts > 1].index)
    if conflicting_statements:
        df_clean = df_clean[~df_clean['statement'].isin(conflicting_statements)]
        
    # 5. Drop exact duplicate statements (keep first)
    df_clean = df_clean.drop_duplicates(subset=['statement'], keep='first')
    
    return df_clean


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """Saves cleaned dataset to target CSV path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved successfully to: {output_path}")


def run_cleaning_pipeline(raw_path: str = "data/raw/Combined Data.csv", 
                          processed_path: str = "data/processed/cleaned_mental_health_data.csv") -> pd.DataFrame:
    """Runs the complete Phase 1 cleaning pipeline and prints diagnostic statistics."""
    print("=" * 60)
    print("PHASE 1 — DATASET CLEANING PIPELINE")
    print("=" * 60)
    
    df_raw = load_raw_data(raw_path)
    raw_stats = inspect_raw_data(df_raw)
    
    print(f"Raw Dataset Loaded: {raw_stats['raw_rows']} rows, {len(raw_stats['columns'])} columns")
    print(f"Columns: {raw_stats['columns']}")
    print(f"Null Counts: {raw_stats['null_counts']}")
    print(f"Exact Full Row Duplicates: {raw_stats['full_row_duplicates']}")
    print(f"Duplicate Statements (Total): {raw_stats['duplicate_statements']}")
    print(f"Statements with Conflicting Labels: {raw_stats['conflicting_statement_count']} unique statements ({raw_stats['conflicting_rows_count']} rows)")
    print("\nRaw Label Distribution:")
    for label, count in raw_stats['label_distribution'].items():
        print(f"  - {label}: {count}")
        
    print("-" * 60)
    print("Executing conservative dataset cleaning...")
    df_clean = clean_dataset(df_raw)
    clean_stats = inspect_raw_data(df_clean)
    
    retained_pct = (clean_stats['raw_rows'] / raw_stats['raw_rows']) * 100
    removed_rows = raw_stats['raw_rows'] - clean_stats['raw_rows']
    
    print(f"\nCleaned Dataset Results:")
    print(f"  - Final Cleaned Rows: {clean_stats['raw_rows']}")
    print(f"  - Total Rows Removed: {removed_rows}")
    print(f"  - Data Retained: {retained_pct:.2f}%")
    print("\nCleaned Label Distribution:")
    for label, count in clean_stats['label_distribution'].items():
        raw_count = raw_stats['label_distribution'].get(label, 0)
        diff = raw_count - count
        pct = (count / clean_stats['raw_rows']) * 100
        print(f"  - {label}: {count} ({pct:.2f}% of clean data, -{diff} removed)")
        
    save_processed_data(df_clean, processed_path)
    
    # Assertions for pipeline validation
    assert df_clean['statement'].isnull().sum() == 0, "Error: Cleaned dataset contains null statements!"
    assert df_clean['status'].isnull().sum() == 0, "Error: Cleaned dataset contains null status!"
    assert df_clean.duplicated(subset=['statement']).sum() == 0, "Error: Cleaned dataset contains duplicate statements!"
    
    conflict_check = df_clean.groupby('statement')['status'].nunique()
    assert (conflict_check > 1).sum() == 0, "Error: Cleaned dataset contains conflicting statement labels!"
    
    print("\nAll pipeline assertions passed successfully!")
    print("=" * 60)
    
    return df_clean


if __name__ == "__main__":
    run_cleaning_pipeline()
