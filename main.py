import argparse
import sys
import logging
from src.pipelines.run_regression import run_regression_pipeline
from src.pipelines.run_forecasting import run_forecasting_pipeline

# Configure global logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Retail Demand Forecasting & Promotional Impact Pipeline")
    parser.add_argument(
        "--task", 
        choices=["regression", "heterogeneity", "forecasting", "prepare"], 
        required=True,
        help="The component of the pipeline to execute. Use 'prepare' to clean data and save to processed/."
    )
    
    args = parser.parse_args()
    
    import os
    transactions_path = "data/raw/wcer.csv"
    products_path = "data/raw/upccer.csv"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    if args.task == "prepare":
        print("Starting Data Preparation Task...")
        try:
            from src.data.load_data import load_data
            from src.data.preprocess import preprocess_data
            from src.data.aggregator import aggregate_to_weekly_chain
            
            print("1. Loading raw data...")
            transactions, products = load_data(transactions_path, products_path)
            
            print("2. Preprocessing & Integrity Audit...")
            df_panel = preprocess_data(transactions, products)
            panel_out = os.path.join(processed_dir, "panel_data.csv")
            df_panel.to_csv(panel_out, index=False)
            print(f"-> Panel data saved to {panel_out}")
            
            print("3. Aggregating to Chain Level...")
            df_chain = aggregate_to_weekly_chain(df_panel)
            chain_out = os.path.join(processed_dir, "chain_data.csv")
            df_chain.to_csv(chain_out, index=False)
            print(f"-> Chain data saved to {chain_out}")
            
            print("Data Preparation Complete.")
        except Exception as e:
            print(f"Error during prepare task: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
    elif args.task in ["regression", "heterogeneity"]:
        print(f"Starting {args.task.capitalize()} Task...")
        try:
            run_regression_pipeline(transactions_path, products_path, task=args.task)
        except Exception as e:
            print(f"Error during {args.task} task: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    elif args.task == "forecasting":
        print("Starting Forecasting Showdown Task...")
        try:
            run_forecasting_pipeline(transactions_path, products_path)
        except Exception as e:
            print(f"Error during forecasting task: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()
