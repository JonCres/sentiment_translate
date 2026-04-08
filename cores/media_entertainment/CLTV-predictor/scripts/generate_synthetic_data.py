import polars as pl
import numpy as np
from datetime import datetime, timedelta
import argparse
from pathlib import Path

def generate_synthetic_avod(user_ids: list, output_path: str, n_records_per_user: int = 5):
    """
    Generates synthetic AVOD (Advertising VOD) data.
    Simulates ad impressions for Basic/Standard tiers.
    """
    print(f"Generating synthetic AVOD data for {len(user_ids)} users...")
    
    n_total = len(user_ids) * n_records_per_user
    
    # Generate random data
    # 30% probability of being an ad-supported user
    is_ad_user = np.random.choice([0, 1], size=len(user_ids), p=[0.7, 0.3])
    ad_users = [uid for uid, is_ad in zip(user_ids, is_ad_user) if is_ad == 1]
    
    if not ad_users:
        print("No ad users generated (check probabilities).")
        return

    # Expand for multiple records
    expanded_users = np.random.choice(ad_users, size=n_total)
    
    # Dates: Last 90 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    random_days = np.random.randint(0, 90, size=n_total)
    dates = [start_date + timedelta(days=int(d)) for d in random_days]
    
    # Ad Metrics
    # 4 mins ads per 60 mins content -> ~6.7% ad load
    # Logic: ad_duration in seconds
    ad_durations = np.random.choice([15, 30, 60], size=n_total, p=[0.4, 0.4, 0.2])
    completed = np.random.choice([0, 1], size=n_total, p=[0.1, 0.9])
    
    # CPMs: $25 US, $3 India (simplified dist)
    cpms = np.random.normal(loc=18, scale=5, size=n_total)
    cpms = np.clip(cpms, 2, 45) # Clip to realistic range

    df = pl.DataFrame({
        "customer_id": expanded_users,
        "event_timestamp": dates,
        "ad_id": [f"AD_{i}" for i in range(n_total)],
        "ad_type": np.random.choice(["pre-roll", "mid-roll"], size=n_total),
        "ad_completed": completed,
        "ad_duration_sec": ad_durations,
        "cpm_usd": cpms
    })
    
    # Ensure dir exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_path)
    print(f"Saved {n_total} AVOD records to {output_path}")


def generate_synthetic_tvod(user_ids: list, output_path: str):
    """
    Generates synthetic TVOD (Transactional VOD) data.
    Simulates PVOD purchases (e.g., $29.99 for blockbusters).
    """
    print(f"Generating synthetic TVOD data...")
    
    # 5% of users make TVOD purchases
    n_purchasers = int(len(user_ids) * 0.05)
    if n_purchasers == 0:
        print("No TVOD purchasers generated.")
        return

    purchasers = np.random.choice(user_ids, size=n_purchasers, replace=False)
    
    # Each purchaser makes 1-3 purchases
    records = []
    
    # Blockbuster titles
    titles = ["Dune: Part Two", "Oppenheimer", "Barbie", "Civil War", "Furiosa"]
    
    for uid in purchasers:
        n_buys = np.random.randint(1, 4)
        for _ in range(n_buys):
            title = np.random.choice(titles)
            price = 29.99 if np.random.random() > 0.2 else 19.99 # Early access vs Rental
            
            # Date
            days_ago = np.random.randint(1, 60)
            txn_date = datetime.now() - timedelta(days=days_ago)
            
            records.append({
                "customer_id": uid,
                "transaction_id": f"TXN_{np.random.randint(100000, 999999)}",
                "transaction_date": txn_date,
                "amount_usd": price,
                "transaction_type": "PVOD",
                "content_title": title
            })
            
    df = pl.DataFrame(records)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(output_path)
    print(f"Saved {len(df)} TVOD records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Synthetic Data for CLTV Predictor")
    parser.add_argument("--subscriptions_path", type=str, required=True, help="Path to existing subscriptions.csv to get user IDs")
    parser.add_argument("--avod_output", type=str, default="data/01_raw/advertising_logs.csv")
    parser.add_argument("--tvod_output", type=str, default="data/01_raw/tvod_transactions.csv")
    
    args = parser.parse_args()
    
    try:
        subs_df = pl.read_csv(args.subscriptions_path, ignore_errors=True)
        user_ids = subs_df["customer_id"].unique().to_list()
        
        if not user_ids:
            print("No user IDs found in subscriptions file.")
            # Fallback: Generate random IDs if file is empty/invalid logic
            user_ids = [f"USER_{i}" for i in range(1000)]
            print(f"Generated {len(user_ids)} fallback user IDs.")
            
    except Exception as e:
        print(f"Error reading subscriptions: {e}")
        user_ids = [f"USER_{i}" for i in range(1000)]
        print(f"Generated {len(user_ids)} fallback user IDs.")

    generate_synthetic_avod(user_ids, args.avod_output)
    generate_synthetic_tvod(user_ids, args.tvod_output)


if __name__ == "__main__":
    main()
