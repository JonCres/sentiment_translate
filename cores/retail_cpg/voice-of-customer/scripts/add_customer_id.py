#!/usr/bin/env python3
"""
Script to add customer_id column to Amazon reviews dataset.
Generates random customer IDs where customers can have multiple reviews.

This script is NOT part of the repository (added to .gitignore).
Run this script to add customer_id to the reviews before analysis.

Usage:
    python scripts/add_customer_id.py
    
    # With custom parameters
    python scripts/add_customer_id.py --customers 500 --seed 123
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def add_customer_ids(
    input_path: str,
    output_path: str = None,
    num_customers: int = 1000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Add customer_id column to reviews dataset.
    
    Uses a power law distribution to simulate realistic customer behavior
    where some customers are prolific reviewers while most write only a few.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file (defaults to input_path)
        num_customers: Number of unique customers to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with customer_id column added
    """
    np.random.seed(seed)
    
    print(f"Reading reviews from: {input_path}")
    df = pd.read_csv(input_path)
    
    # Generate customer IDs
    customer_ids = [f"customer_{i:05d}" for i in range(num_customers)]
    
    # Use power law distribution for customer assignment
    # This creates a realistic distribution where:
    # - Some customers are very active (many reviews)
    # - Most customers write only 1-2 reviews
    weights = np.power(np.arange(1, num_customers + 1), -0.7)
    weights = weights / weights.sum()
    
    # Assign customer IDs to reviews
    df['customer_id'] = np.random.choice(
        customer_ids, 
        size=len(df), 
        p=weights
    )
    
    # Save to output path
    output_path = output_path or input_path
    df.to_csv(output_path, index=False)
    
    # Print summary statistics
    print(f"\n{'='*50}")
    print("Customer ID Generation Summary")
    print(f"{'='*50}")
    print(f"Total reviews: {len(df)}")
    print(f"Unique customers: {df['customer_id'].nunique()}")
    print(f"Average reviews per customer: {len(df) / df['customer_id'].nunique():.2f}")
    
    # Show distribution of reviews per customer
    reviews_per_customer = df['customer_id'].value_counts()
    print(f"\nReviews per customer distribution:")
    print(f"  Min: {reviews_per_customer.min()}")
    print(f"  Max: {reviews_per_customer.max()}")
    print(f"  Median: {reviews_per_customer.median():.0f}")
    print(f"  Mean: {reviews_per_customer.mean():.2f}")
    
    # Top 5 most active customers
    print(f"\nTop 5 most active customers:")
    for customer_id, count in reviews_per_customer.head(5).items():
        print(f"  {customer_id}: {count} reviews")
    
    print(f"\nSaved to: {output_path}")
    print(f"{'='*50}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Add customer_id column to Amazon reviews dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        default='data/01_raw/amazon_reviews.csv',
        help="Input CSV file path (default: data/01_raw/amazon_reviews.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default='data/01_raw/amazon_reviews.csv',
        help="Output CSV file path (default: same as input)"
    )
    parser.add_argument(
        "--customers",
        type=int,
        default=1000,
        help="Number of unique customers to generate (default: 1000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()

    add_customer_ids(
        input_path=args.input,
        output_path=args.output,
        num_customers=args.customers,
        seed=args.seed
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
