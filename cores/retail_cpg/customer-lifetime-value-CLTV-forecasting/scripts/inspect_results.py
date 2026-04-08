from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from pathlib import Path
import polars as pl

def inspect_results():
    project_path = Path.cwd()
    bootstrap_project(project_path)
    with KedroSession.create(project_path=project_path) as session:
        context = session.load_context()
        catalog = context.catalog
        
        print("\n--- CLTV Predictions ---")
        df_pred = catalog.load("cltv_predictions")
        print(df_pred.head())
        print(f"Total predictions: {len(df_pred)}")
        print(f"Columns: {df_pred.columns}")
        
        print("\n--- PII Masked Data ---")
        df_masked = catalog.load("masked_data")
        print(df_masked.head())
        # Check if customer_id is hashed (it was mapped from Customer ID)
        # Check if customer_feedback is masked
        print(f"Masked columns: {df_masked.columns}")

if __name__ == "__main__":
    inspect_results()
