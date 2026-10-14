"""
PHM 2018 Dataset Inspection Script
Analyzes the actual dataset structure without assumptions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

DATASET_PATH = Path("data/raw/uploads/phm_data_challenge_2018")

def inspect_dataset():
    """Inspect PHM 2018 dataset and report actual structure."""
    
    report = {
        "dataset_name": "PHM 2018 Ion Mill Etch Tool",
        "dataset_path": str(DATASET_PATH),
        "files": [],
        "summary": {}
    }
    
    # Find all CSV files
    csv_files = list(DATASET_PATH.glob("**/*.csv"))
    report["summary"]["total_files"] = len(csv_files)
    
    for csv_file in csv_files:
        file_info = {
            "path": str(csv_file.relative_to(DATASET_PATH)),
            "size_bytes": csv_file.stat().st_size,
            "rows": 0,
            "columns": [],
            "column_types": {},
            "missing_values": {},
            "sample_data": None
        }
        
        try:
            df = pd.read_csv(csv_file)
            file_info["rows"] = len(df)
            file_info["columns"] = df.columns.tolist()
            
            # Column types
            for col in df.columns:
                file_info["column_types"][col] = str(df[col].dtype)
            
            # Missing values
            missing = df.isnull().sum()
            for col in df.columns:
                if missing[col] > 0:
                    file_info["missing_values"][col] = int(missing[col])
            
            # Sample first few rows
            file_info["sample_data"] = df.head(3).to_dict(orient="records")
            
            # Check for unique identifiers
            if "Tool" in df.columns:
                file_info["unique_tools"] = df["Tool"].unique().tolist()
            if "recipe" in df.columns:
                file_info["unique_recipes"] = df["recipe"].unique().tolist()
            if "stage" in df.columns:
                file_info["unique_stages"] = df["stage"].unique().tolist()
                
        except Exception as e:
            file_info["error"] = str(e)
        
        report["files"].append(file_info)
    
    # Save report
    report_path = Path("results/dataset_inspection_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Dataset inspection complete. Report saved to {report_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("PHM 2018 DATASET INSPECTION SUMMARY")
    print("="*60)
    print(f"Total files: {report['summary']['total_files']}")
    print(f"\nFiles found:")
    for file_info in report["files"]:
        print(f"  - {file_info['path']}")
        print(f"    Rows: {file_info['rows']:,}")
        print(f"    Columns: {len(file_info['columns'])}")
        print(f"    Columns: {', '.join(file_info['columns'][:5])}...")
        if file_info.get("unique_tools"):
            print(f"    Tools: {file_info['unique_tools']}")
        if file_info.get("unique_recipes"):
            print(f"    Recipes: {file_info['unique_recipes']}")
        print()
    
    return report

if __name__ == "__main__":
    inspect_dataset()
