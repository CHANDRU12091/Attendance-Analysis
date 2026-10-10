"""
Test Suite for Attendify
Tests data processing, regression modeling, and visualizations.
"""

import sys
import pandas as pd
import numpy as np

from modules.data_processing import (
    load_dataset,
    save_dataset,
    add_student_record,
    clean_dataset,
    calculate_numpy_statistics,
    get_attendance_category_counts,
    get_below_threshold_students
)
from modules.regression import perform_linear_regression, predict_final_marks
from modules.visualizations import (
    create_attendance_distribution_chart,
    create_regression_scatter_chart,
    create_study_hours_vs_marks_chart,
    create_avg_marks_by_category_chart,
    create_top_students_attendance_chart
)


def run_all_tests():
    print("Running Attendify Test Suite...")

    # 1. Dataset Loading
    df, err = load_dataset('attendance.csv')
    assert err is None, f"Error loading CSV: {err}"
    assert len(df) >= 10, f"Expected at least 10 rows, got {len(df)}"
    print(f"[PASS] 1. Loaded attendance.csv with {len(df)} records.")

    # 2. NumPy Statistics
    stats = calculate_numpy_statistics(df)
    print("       Computed NumPy Statistics:")
    for k, v in stats.items():
        print(f"         - {k}: {v}")
    assert 'mean_attendance' in stats
    assert 'correlation' in stats
    print("[PASS] 2. NumPy statistics verified.")

    # 3. Regression Analysis
    reg_result, reg_err = perform_linear_regression(df)
    assert reg_err is None, f"Regression error: {reg_err}"
    print(f"       Equation: {reg_result['equation']}")
    print(f"       R²: {reg_result['r2_score']}, Correlation: {reg_result['correlation']}")
    pred = predict_final_marks(reg_result['slope'], reg_result['intercept'], 85.0)
    print(f"       Prediction for 85% attendance: {pred}")
    print("[PASS] 3. Regression module verified.")

    # 4. Matplotlib Visualizations
    fig1 = create_attendance_distribution_chart(df)
    fig2 = create_regression_scatter_chart(df, reg_result)
    fig3 = create_study_hours_vs_marks_chart(df)
    fig4 = create_avg_marks_by_category_chart(df)
    fig5 = create_top_students_attendance_chart(df)
    assert fig1 is not None and fig2 is not None and fig3 is not None and fig4 is not None and fig5 is not None
    print("[PASS] 4. All 5 Matplotlib figures created successfully.")

    # 5. Add Student Validation & Automatic Attendance Calculation
    test_new_student = {
        'Student_ID': 'S099',
        'Student_Name': 'Test Student',
        'Department': 'CSE',
        'Year': 2,
        'Total_Classes': 50,
        'Classes_Attended': 45,
        'Study_Hours': 7.0,
        'Internal_Marks': 80.0,
        'Final_Marks': 85.0
    }
    updated_df, success, msg = add_student_record(df, test_new_student)
    assert success, f"Failed to add student: {msg}"
    assert updated_df.loc[updated_df['Student_ID'] == 'S099', 'Attendance'].iloc[0] == 90.0
    print("[PASS] 5. Automatic attendance calculation verified (45/50 = 90.0%).")

    # 6. Duplicate Prevention
    _, dup_success, dup_msg = add_student_record(
        df,
        {
            'Student_ID': 'S001',
            'Student_Name': 'Duplicate',
            'Department': 'CSE',
            'Year': 2,
            'Total_Classes': 50,
            'Classes_Attended': 40,
            'Study_Hours': 5,
            'Internal_Marks': 50,
            'Final_Marks': 50
        }
    )
    assert not dup_success, "Duplicate ID was unexpectedly allowed."
    print(f"[PASS] 6. Duplicate ID blocked properly ({dup_msg}).")

    # 7. Data Cleaning with Pandas
    dirty_df = df.copy()
    dirty_df.loc[0, 'Study_Hours'] = np.nan
    dirty_df.loc[1, 'Internal_Marks'] = np.nan
    dirty_df = pd.concat([dirty_df, dirty_df.iloc[[0]]], ignore_index=True)
    cleaned_df, report = clean_dataset(dirty_df)
    assert report['total_missing_before'] >= 2
    assert report['duplicates_removed'] >= 1
    assert report['missing_values_after'] == 0
    print("[PASS] 7. Pandas data cleaning (missing imputation + duplicate removal) verified.")

    print("\n========================================================")
    print("ALL UNIT AND INTEGRATION CHECKS PASSED WITH 100% SUCCESS!")
    print("========================================================")


if __name__ == "__main__":
    run_all_tests()
