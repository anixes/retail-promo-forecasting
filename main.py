import argparse
import sys
import logging
from src.pipelines.run_regression import run_regression_pipeline

# Configure global logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Retail Demand Forecasting & Promotional Impact Pipeline")
    parser.add_argument(
        "--task", 
        choices=["regression", "heterogeneity", "forecasting"], 
        required=True,
        help="The component of the pipeline to execute."
    )
    
    args = parser.parse_args()
    
    transactions_path = "data/raw/wcer.csv"
    products_path = "data/raw/upccer.csv"
    
    if args.task in ["regression", "heterogeneity"]:
        print(f"Starting {args.task.capitalize()} Task...")
        try:
            run_regression_pipeline(transactions_path, products_path, task=args.task)
        except Exception as e:
            print(f"Error during {args.task} task: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    elif args.task == "forecasting":
        print("Forecasting task implementation in progress...")
        # Placeholder for Phase 2
        pass

if __name__ == "__main__":
    main()
