#!/usr/bin/env python3
"""
Sample script to show Excel report structure
"""
import pandas as pd

try:
    # Read the Excel file
    excel_file = "testing_report_20250805_202321.xlsx"
    
    # Read the main test results sheet
    df_results = pd.read_excel(excel_file, sheet_name='Test Results')
    
    # Read the summary sheet
    df_summary = pd.read_excel(excel_file, sheet_name='Ringkasan')
    
    print("=== STRUKTUR EXCEL REPORT ===\n")
    
    print("Sheet 1: Test Results")
    print("Kolom yang tersedia:")
    for i, col in enumerate(df_results.columns, 1):
        print(f"{i}. {col}")
    
    print(f"\nTotal baris data: {len(df_results)}")
    
    print("\n=== SAMPLE DATA (5 baris pertama) ===")
    print(df_results.head().to_string(index=False))
    
    print("\n\nSheet 2: Ringkasan")
    print(df_summary.to_string(index=False))
    
except Exception as e:
    print(f"Error: {e}")
    print("Make sure openpyxl is installed: pip install openpyxl")