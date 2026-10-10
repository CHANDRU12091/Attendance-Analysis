"""
Data Processing Module for Attendify
Handles dataset loading, validation, cleaning with Pandas, and numerical statistics with NumPy.
"""

from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    'Student_ID',
    'Student_Name',
    'Department',
    'Year',
    'Total_Classes',
    'Classes_Attended',
    'Attendance',
    'Study_Hours',
    'Internal_Marks',
    'Final_Marks'
]

NUMERIC_COLUMNS = [
    'Total_Classes',
    'Classes_Attended',
    'Attendance',
    'Study_Hours',
    'Internal_Marks',
    'Final_Marks'
]


def load_dataset(filepath_or_buffer) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load attendance CSV dataset and validate required schema.
    Returns (DataFrame, error_message).
    """
    try:
        df = pd.read_csv(filepath_or_buffer)
        if df.empty:
            return None, "The uploaded dataset is empty."
        
        # Check required columns
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            return None, f"Dataset is missing required columns: {', '.join(missing_cols)}"
        
        # Ensure numeric columns are properly typed
        for col in NUMERIC_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Re-calculate Attendance to ensure consistency
        mask_valid = (df['Total_Classes'] > 0) & (df['Total_Classes'].notnull())
        df.loc[mask_valid, 'Attendance'] = (
            (df.loc[mask_valid, 'Classes_Attended'] / df.loc[mask_valid, 'Total_Classes']) * 100
        ).round(2)
        
        # Add category column
        df['Attendance_Category'] = df['Attendance'].apply(categorize_attendance)
        
        return df, None
    except Exception as e:
        return None, f"Error loading dataset: {str(e)}"


def save_dataset(df: pd.DataFrame, filepath: str) -> bool:
    """Save DataFrame to CSV file without extra computed columns."""
    try:
        cols_to_save = [c for c in REQUIRED_COLUMNS if c in df.columns]
        df[cols_to_save].to_csv(filepath, index=False)
        return True
    except Exception:
        return False


def categorize_attendance(attendance: float) -> str:
    """
    Categorize attendance percentage into 4 standard academic tiers:
    - 90–100%  -> Excellent
    - 75–89%   -> Good
    - 60–74%   -> Average
    - Below 60% -> Poor
    """
    if pd.isna(attendance):
        return "Unknown"
    if attendance >= 90.0:
        return "Excellent"
    elif attendance >= 75.0:
        return "Good"
    elif attendance >= 60.0:
        return "Average"
    else:
        return "Poor"


def validate_student_input(
    student_id: str,
    name: str,
    department: str,
    year: int,
    total_classes: int,
    classes_attended: int,
    study_hours: float,
    internal_marks: float,
    final_marks: float,
    existing_ids: list
) -> Tuple[bool, str]:
    """Validate all student inputs before adding record."""
    if not student_id or not student_id.strip():
        return False, "Student ID cannot be empty."
    
    clean_id = student_id.strip()
    if clean_id in [str(x).strip() for x in existing_ids]:
        return False, f"Student ID '{clean_id}' already exists in the dataset. Duplicate IDs are not allowed."
        
    if not name or not name.strip():
        return False, "Student Name cannot be empty."
        
    if not department or not department.strip():
        return False, "Department must be selected."
        
    if year not in [1, 2, 3, 4]:
        return False, "Year must be between 1 and 4."
        
    if total_classes <= 0:
        return False, "Total classes must be greater than zero."
        
    if classes_attended < 0:
        return False, "Classes attended cannot be negative."
        
    if classes_attended > total_classes:
        return False, f"Classes attended ({classes_attended}) cannot exceed Total classes ({total_classes})."
        
    if study_hours < 0:
        return False, "Study hours cannot be negative."
        
    if not (0 <= internal_marks <= 100):
        return False, "Internal marks must be between 0 and 100."
        
    if not (0 <= final_marks <= 100):
        return False, "Final marks must be between 0 and 100."
        
    return True, "Valid"


def add_student_record(
    df: pd.DataFrame,
    student_data: Dict[str, Any]
) -> Tuple[pd.DataFrame, bool, str]:
    """
    Add a new student record to the DataFrame and automatically calculate attendance.
    """
    valid, msg = validate_student_input(
        student_id=str(student_data['Student_ID']),
        name=str(student_data['Student_Name']),
        department=str(student_data['Department']),
        year=int(student_data['Year']),
        total_classes=int(student_data['Total_Classes']),
        classes_attended=int(student_data['Classes_Attended']),
        study_hours=float(student_data['Study_Hours']),
        internal_marks=float(student_data['Internal_Marks']),
        final_marks=float(student_data['Final_Marks']),
        existing_ids=df['Student_ID'].tolist() if 'Student_ID' in df.columns else []
    )
    
    if not valid:
        return df, False, msg
        
    # Automatic attendance calculation: (Classes Attended / Total Classes) * 100
    total_cls = int(student_data['Total_Classes'])
    attended_cls = int(student_data['Classes_Attended'])
    attendance_pct = round((attended_cls / total_cls) * 100, 2)
    
    new_row = {
        'Student_ID': str(student_data['Student_ID']).strip(),
        'Student_Name': str(student_data['Student_Name']).strip(),
        'Department': str(student_data['Department']).strip(),
        'Year': int(student_data['Year']),
        'Total_Classes': total_cls,
        'Classes_Attended': attended_cls,
        'Attendance': attendance_pct,
        'Study_Hours': float(student_data['Study_Hours']),
        'Internal_Marks': float(student_data['Internal_Marks']),
        'Final_Marks': float(student_data['Final_Marks']),
        'Attendance_Category': categorize_attendance(attendance_pct)
    }
    
    updated_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return updated_df, True, "Student added successfully!"


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform comprehensive data cleaning using Pandas:
    1. Detect missing values (df.isnull().sum())
    2. Handle missing numerical values using column mean: df[col] = df[col].fillna(df[col].mean())
    3. Handle missing categorical values using column mode: df[col].fillna(mode)
    4. Detect and drop duplicate records (df.drop_duplicates())
    5. Detect & resolve invalid numeric ranges (classes attended > total classes, marks not in [0, 100])
    6. Type conversion
    Returns cleaned DataFrame and cleaning audit report dictionary.
    """
    original_count = len(df)
    missing_by_column_before = df.isnull().sum().to_dict()
    total_missing_before = int(df.isnull().sum().sum())
    duplicates_before = int(df.duplicated(subset=['Student_ID']).sum()) if 'Student_ID' in df.columns else int(df.duplicated().sum())
    
    cleaned_df = df.copy()
    
    # 1. Drop duplicates based on Student_ID if available, else entire row
    if 'Student_ID' in cleaned_df.columns:
        cleaned_df = cleaned_df.drop_duplicates(subset=['Student_ID'], keep='first')
    else:
        cleaned_df = cleaned_df.drop_duplicates()
        
    duplicates_removed = original_count - len(cleaned_df)
    
    # 2. Impute numerical columns using column mean
    numerical_cols = [c for c in NUMERIC_COLUMNS if c in cleaned_df.columns]
    for col in numerical_cols:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        mean_val = cleaned_df[col].mean()
        if pd.isna(mean_val):
            mean_val = 0.0
        cleaned_df[col] = cleaned_df[col].fillna(round(mean_val, 2))
        
    # 3. Impute categorical columns using mode
    cat_cols = ['Student_ID', 'Student_Name', 'Department', 'Year']
    for col in cat_cols:
        if col in cleaned_df.columns:
            mode_series = cleaned_df[col].mode()
            mode_val = mode_series.iloc[0] if not mode_series.empty else "N/A"
            cleaned_df[col] = cleaned_df[col].fillna(mode_val)
            
    # 4. Outlier & Sanity checks
    if 'Total_Classes' in cleaned_df.columns and 'Classes_Attended' in cleaned_df.columns:
        # Total classes must be at least 1
        cleaned_df['Total_Classes'] = cleaned_df['Total_Classes'].apply(lambda x: max(1, int(x)))
        # Classes attended cannot exceed total classes
        cleaned_df['Classes_Attended'] = np.minimum(
            cleaned_df['Classes_Attended'], cleaned_df['Total_Classes']
        )
        cleaned_df['Classes_Attended'] = np.maximum(0, cleaned_df['Classes_Attended'])
        # Re-calculate Attendance
        cleaned_df['Attendance'] = (
            (cleaned_df['Classes_Attended'] / cleaned_df['Total_Classes']) * 100
        ).round(2)
        
    # 5. Clip Marks between 0 and 100, Study hours >= 0
    if 'Internal_Marks' in cleaned_df.columns:
        cleaned_df['Internal_Marks'] = cleaned_df['Internal_Marks'].clip(0, 100).round(1)
    if 'Final_Marks' in cleaned_df.columns:
        cleaned_df['Final_Marks'] = cleaned_df['Final_Marks'].clip(0, 100).round(1)
    if 'Study_Hours' in cleaned_df.columns:
        cleaned_df['Study_Hours'] = cleaned_df['Study_Hours'].clip(lower=0).round(1)
        
    # Update category
    cleaned_df['Attendance_Category'] = cleaned_df['Attendance'].apply(categorize_attendance)
    
    total_missing_after = int(cleaned_df.isnull().sum().sum())
    duplicates_after = int(cleaned_df.duplicated(subset=['Student_ID']).sum()) if 'Student_ID' in cleaned_df.columns else 0
    
    audit_report = {
        'original_records': original_count,
        'missing_values_before': missing_by_column_before,
        'total_missing_before': total_missing_before,
        'duplicates_before': duplicates_before,
        'duplicates_removed': duplicates_removed,
        'cleaned_records': len(cleaned_df),
        'missing_values_after': total_missing_after,
        'duplicates_after': duplicates_after
    }
    
    return cleaned_df, audit_report


def calculate_numpy_statistics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute core statistical metrics using NumPy explicitly:
    - np.mean()
    - np.median()
    - np.std()
    - np.min()
    - np.max()
    - np.corrcoef()
    """
    if df.empty or 'Attendance' not in df.columns or 'Final_Marks' not in df.columns:
        return {}
        
    att_array = np.array(df['Attendance'].dropna(), dtype=float)
    marks_array = np.array(df['Final_Marks'].dropna(), dtype=float)
    
    if len(att_array) == 0 or len(marks_array) == 0:
        return {}
        
    mean_att = float(np.mean(att_array))
    median_att = float(np.median(att_array))
    std_att = float(np.std(att_array, ddof=0))
    min_att = float(np.min(att_array))
    max_att = float(np.max(att_array))
    
    mean_marks = float(np.mean(marks_array))
    median_marks = float(np.median(marks_array))
    std_marks = float(np.std(marks_array, ddof=0))
    min_marks = float(np.min(marks_array))
    max_marks = float(np.max(marks_array))
    
    # Pearson correlation coefficient matrix: np.corrcoef(x, y)
    if len(att_array) > 1 and len(att_array) == len(marks_array):
        # ensure no zero std
        if np.std(att_array) > 0 and np.std(marks_array) > 0:
            corr_matrix = np.corrcoef(att_array, marks_array)
            correlation = float(corr_matrix[0, 1])
        else:
            correlation = 0.0
    else:
        correlation = 0.0
        
    return {
        'mean_attendance': round(mean_att, 2),
        'median_attendance': round(median_att, 2),
        'std_attendance': round(std_att, 2),
        'min_attendance': round(min_att, 2),
        'max_attendance': round(max_att, 2),
        'mean_final_marks': round(mean_marks, 2),
        'median_final_marks': round(median_marks, 2),
        'std_final_marks': round(std_marks, 2),
        'min_final_marks': round(min_marks, 2),
        'max_final_marks': round(max_marks, 2),
        'correlation': round(correlation, 4)
    }


def get_attendance_category_counts(df: pd.DataFrame) -> Dict[str, int]:
    """Get count of students in each attendance category."""
    categories = ['Excellent', 'Good', 'Average', 'Poor']
    if 'Attendance_Category' not in df.columns:
        return {cat: 0 for cat in categories}
    counts = df['Attendance_Category'].value_counts().to_dict()
    return {cat: counts.get(cat, 0) for cat in categories}


def get_below_threshold_students(df: pd.DataFrame, threshold: float = 75.0) -> pd.DataFrame:
    """Return students whose attendance is below threshold (default 75%)."""
    if 'Attendance' not in df.columns:
        return pd.DataFrame()
    return df[df['Attendance'] < threshold].sort_values(by='Attendance', ascending=True)


def get_excellent_students(df: pd.DataFrame, threshold: float = 90.0) -> pd.DataFrame:
    """Return students whose attendance is >= threshold (default 90%)."""
    if 'Attendance' not in df.columns:
        return pd.DataFrame()
    return df[df['Attendance'] >= threshold].sort_values(by='Attendance', ascending=False)
