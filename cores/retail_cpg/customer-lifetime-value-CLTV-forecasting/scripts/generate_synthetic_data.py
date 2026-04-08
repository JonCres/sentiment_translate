import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(n_customers=100, n_transactions=1000):
    customer_ids = [f"CUST_{i}" for i in range(n_customers)]
    channels = ["Paid Search", "Organic", "Social Media", "Email", "Direct"]
    
    data = []
    start_date = datetime(2010, 1, 1)
    
    for _ in range(n_transactions):
        customer_id = np.random.choice(customer_ids)
        days_offset = np.random.randint(0, 365)
        invoice_date = start_date + timedelta(days=days_offset)
        
        # Non-contractual logic: Poisson-like frequency
        quantity = np.random.randint(1, 10)
        price = np.random.uniform(10.0, 100.0)
        
        data.append({
            "Customer ID": customer_id,
            "Customer Name": "John Doe",
            "InvoiceDate": invoice_date.strftime("%Y-%m-%d %H:%M:%S"),
            "Quantity": quantity,
            "Price": price,
            "Country": "United Kingdom",
            "acquisition_channel": np.random.choice(channels),
            "return_flag": 0,
            "customer_feedback": f"My email is {customer_id.lower()}@example.com"
        })
        
    df = pd.DataFrame(data)
    df.to_csv("data/01_raw/online_retail_II.csv", index=False)
    print("Generated synthetic data in data/01_raw/online_retail_II.csv")

if __name__ == "__main__":
    import os
    os.makedirs("data/01_raw", exist_ok=True)
    generate_synthetic_data()
