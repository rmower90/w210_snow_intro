"""
Module: preprocessing

This module handles the preprocessing of raw SWE and atmospheric data.
"""

def preprocess_data(raw_data: dict) -> dict:
    """
    Preprocess the raw input data.

    Args:
        raw_data (dict): Raw input data.

    Returns:
        dict: Cleaned and preprocessed data.
    """
    # Example: Convert all values to float.
    processed = {k: float(v) for k, v in raw_data.items()}
    return processed

def normalize_data(data: dict) -> dict:
    """
    Normalize the data using min-max normalization.

    Args:
        data (dict): Dictionary of numerical values.

    Returns:
        dict: Normalized data.
    """
    values = list(data.values())
    min_val = min(values)
    max_val = max(values)
    normalized = {k: (v - min_val) / (max_val - min_val) for k, v in data.items()}
    return normalized
