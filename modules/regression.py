"""
Regression Analysis Module for Attendify
Performs Simple Linear Regression using NumPy (polyfit) and calculates R², correlation,
predictions, and academic interpretations.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd


def perform_linear_regression(
    df: pd.DataFrame,
    x_col: str = 'Attendance',
    y_col: str = 'Final_Marks'
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Perform Simple Linear Regression: y = m*x + c
    Independent Variable (X): Attendance (%)
    Dependent Variable (Y): Final Marks

    Uses NumPy polyfit(x, y, 1).
    Calculates:
      - slope (m)
      - intercept (c)
      - correlation coefficient (r)
      - coefficient of determination (R²)
      - regression equation string
      - model interpretation
    """
    if df.empty:
        return None, "The dataset is empty. Cannot perform regression."

    if x_col not in df.columns or y_col not in df.columns:
        return None, f"Required columns '{x_col}' and '{y_col}' are missing."

    valid_data = df[[x_col, y_col]].dropna()
    if len(valid_data) < 2:
        return None, "At least 2 data points are required to compute linear regression."

    x = np.array(valid_data[x_col], dtype=float)
    y = np.array(valid_data[y_col], dtype=float)

    # Check for zero variance in X
    if np.std(x) == 0:
        return None, "Attendance values have zero variance (all values are identical). Cannot fit a unique regression line."

    # 1. Fit linear model using NumPy's polyfit (degree 1)
    slope, intercept = np.polyfit(x, y, 1)

    # 2. Predicted values
    y_pred = slope * x + intercept

    # 3. Sum of Squared Total (SS_tot) and Sum of Squared Residuals (SS_res)
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)

    # 4. R² Score (Coefficient of Determination)
    if ss_tot > 0:
        r2 = 1.0 - (ss_res / ss_tot)
    else:
        r2 = 1.0 if ss_res == 0 else 0.0
    r2 = max(0.0, min(1.0, float(r2)))

    # 5. Pearson Correlation Coefficient using np.corrcoef
    if np.std(x) > 0 and np.std(y) > 0:
        corr_matrix = np.corrcoef(x, y)
        correlation = float(corr_matrix[0, 1])
    else:
        correlation = 0.0

    # 6. Format equation string
    sign = "+" if intercept >= 0 else "-"
    equation_str = f"Final Marks = {slope:.2f} * Attendance {sign} {abs(intercept):.2f}"

    # 7. Generate scientific interpretation
    interpretation = generate_regression_interpretation(slope, correlation, r2)

    result = {
        'x': x,
        'y': y,
        'y_pred': y_pred,
        'slope': float(slope),
        'intercept': float(intercept),
        'r2_score': round(float(r2), 4),
        'correlation': round(float(correlation), 4),
        'equation': equation_str,
        'interpretation': interpretation,
        'sample_size': len(x)
    }

    return result, None


def predict_final_marks(slope: float, intercept: float, attendance_value: float) -> float:
    """
    Predict Final Marks given an Attendance percentage using the fitted model:
    Final Marks = slope * Attendance + intercept
    Clipped between 0 and 100 as realistic bounds.
    """
    predicted = slope * attendance_value + intercept
    return round(float(np.clip(predicted, 0, 100)), 2)


def generate_regression_interpretation(slope: float, correlation: float, r2: float) -> Dict[str, str]:
    """
    Produce a beginner-friendly, academically sound data-science explanation
    distinguishing correlation, regression, prediction, and causation.
    """
    abs_corr = abs(correlation)
    if abs_corr >= 0.8:
        strength = "strong"
    elif abs_corr >= 0.5:
        strength = "moderate"
    elif abs_corr >= 0.3:
        strength = "weak"
    else:
        strength = "very weak / negligible"

    direction = "positive" if correlation > 0 else ("negative" if correlation < 0 else "neutral")

    summary = (
        f"There is a {strength} {direction} correlation (r = {correlation:.2f}) "
        f"between attendance and final marks in this dataset."
    )

    if correlation > 0:
        slope_explanation = (
            f"For every 10% increase in attendance, the model estimates an expected increase "
            f"of approximately {abs(slope * 10):.1f} marks in final examination score."
        )
    elif correlation < 0:
        slope_explanation = (
            f"For every 10% increase in attendance, the model estimates an unexpected decrease "
            f"of approximately {abs(slope * 10):.1f} marks."
        )
    else:
        slope_explanation = "Attendance shows no linear association with final marks in this dataset."

    r2_percent = r2 * 100
    r2_explanation = (
        f"The R² score of {r2:.4f} ({r2_percent:.1f}%) indicates that approximately "
        f"{r2_percent:.1f}% of the variance in final marks can be explained by attendance "
        f"using this linear model. The remaining {100 - r2_percent:.1f}% is influenced by other "
        f"factors such as independent study hours, prior knowledge, internal assessments, and coursework."
    )

    causality_note = (
        "Important Data Science Note: This analysis establishes a statistical association "
        "(correlation and regression line), but DOES NOT prove direct causation. While attending classes "
        "gives students access to lectures and guidance, higher marks also depend on consistent study habits, "
        "practice, and comprehension."
    )

    return {
        'summary': summary,
        'slope_explanation': slope_explanation,
        'r2_explanation': r2_explanation,
        'causality_note': causality_note
    }
