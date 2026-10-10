"""
Visualizations Module for Attendify
Creates professional, publication-quality academic charts using Matplotlib.
Includes 5 essential charts:
1. Attendance Distribution (Bar Chart)
2. Attendance vs Final Marks with Regression Line (Scatter + Line)
3. Study Hours vs Final Marks (Scatter Plot)
4. Average Marks by Attendance Category (Bar Chart)
5. Top 10 Students by Attendance (Horizontal Bar Chart)
"""

from typing import Dict, Any, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend suitable for web servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Consistent Academic Styling Palette
COLORS = {
    'primary': '#1E40AF',       # Academic Royal Blue
    'secondary': '#0D9488',     # Modern Teal
    'accent': '#6366F1',        # Indigo
    'warning': '#F59E0B',       # Amber
    'danger': '#EF4444',        # Red
    'success': '#10B981',       # Emerald Green
    'neutral_dark': '#1F2937',  # Slate 800
    'neutral_light': '#F3F4F6', # Slate 100
    'grid': '#E5E7EB',          # Light Gray
    'excellent': '#10B981',     # Green
    'good': '#3B82F6',          # Blue
    'average': '#F59E0B',       # Amber
    'poor': '#EF4444'           # Red
}


def create_attendance_distribution_chart(df: pd.DataFrame) -> plt.Figure:
    """
    Chart 1 — Attendance Distribution
    Bar chart showing the number of students in each attendance tier.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    categories = ['Excellent\n(90-100%)', 'Good\n(75-89%)', 'Average\n(60-74%)', 'Poor\n(<60%)']
    raw_cats = ['Excellent', 'Good', 'Average', 'Poor']
    colors = [COLORS['excellent'], COLORS['good'], COLORS['average'], COLORS['poor']]
    
    cat_counts = df['Attendance_Category'].value_counts() if 'Attendance_Category' in df.columns else {}
    counts = [int(cat_counts.get(c, 0)) for c in raw_cats]
    
    bars = ax.bar(categories, counts, color=colors, width=0.55, edgecolor='#1E293B', linewidth=0.8)
    
    # Add count labels above bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold', color='#111827')
                    
    ax.set_title('Attendance Distribution by Academic Tier', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Attendance Tier', fontsize=12, labelpad=10, fontweight='bold')
    ax.set_ylabel('Number of Students', fontsize=12, labelpad=10, fontweight='bold')
    
    max_y = max(counts) if counts and max(counts) > 0 else 5
    ax.set_ylim(0, max_y + max(2, int(max_y * 0.2)))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(axis='y', linestyle='--', alpha=0.5, color=COLORS['grid'])
    ax.set_axisbelow(True)
    
    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    return fig


def create_regression_scatter_chart(
    df: pd.DataFrame,
    regression_info: Optional[Dict[str, Any]] = None
) -> plt.Figure:
    """
    Chart 2 — Attendance vs Final Marks with Regression Line
    Scatter plot with linear regression line, labels, legend, and grid.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    x = df['Attendance'].values
    y = df['Final_Marks'].values
    
    # Scatter points
    ax.scatter(x, y, color=COLORS['primary'], alpha=0.85, s=65, edgecolors='white',
               linewidth=1.2, label='Student Observations', zorder=3)
               
    # Plot Regression Line if available
    if regression_info is not None:
        slope = regression_info['slope']
        intercept = regression_info['intercept']
        r2 = regression_info['r2_score']
        
        # Continuous line for smooth plotting
        x_line = np.linspace(min(x) - 2, max(x) + 2, 100)
        y_line = slope * x_line + intercept
        
        eq_label = f"Fit: y = {slope:.2f}x + {intercept:.2f} (R² = {r2:.2f})"
        ax.plot(x_line, y_line, color=COLORS['danger'], linestyle='-', linewidth=2.5,
                label=eq_label, zorder=4)
                
    # 75% Attendance minimum threshold benchmark line
    ax.axvline(75, color=COLORS['warning'], linestyle=':', linewidth=1.8,
               label='75% Minimum Attendance Threshold', zorder=2)
               
    ax.set_title('Attendance vs Final Marks with Regression Line', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Attendance (%)', fontsize=12, labelpad=10, fontweight='bold')
    ax.set_ylabel('Final Marks (0 - 100)', fontsize=12, labelpad=10, fontweight='bold')
    
    ax.set_xlim(max(0, min(x) - 5), min(105, max(x) + 5))
    ax.set_ylim(max(0, min(y) - 10), min(105, max(y) + 5))
    
    ax.grid(True, linestyle='--', alpha=0.5, color=COLORS['grid'])
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=10)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    return fig


def create_study_hours_vs_marks_chart(df: pd.DataFrame) -> plt.Figure:
    """
    Chart 3 — Study Hours vs Final Marks
    Scatter plot analyzing relationship between independent study hours and final score.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    hours = df['Study_Hours'].values
    marks = df['Final_Marks'].values
    
    scatter = ax.scatter(hours, marks, c=df['Attendance'], cmap='Blues', s=70,
                         edgecolors='#1E293B', linewidth=1.0, alpha=0.9, zorder=3)
                         
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Attendance (%)', fontsize=10, fontweight='bold')
    
    # Fit simple trendline if variance exists
    if len(hours) > 1 and np.std(hours) > 0:
        poly_fit = np.polyfit(hours, marks, 1)
        x_trend = np.linspace(min(hours), max(hours), 50)
        y_trend = poly_fit[0] * x_trend + poly_fit[1]
        ax.plot(x_trend, y_trend, color=COLORS['secondary'], linestyle='--', linewidth=2,
                label=f'Trendline (m = {poly_fit[0]:.2f})', zorder=2)
        ax.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=10)
        
    ax.set_title('Study Hours vs Final Marks', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Independent Study Hours per Week', fontsize=12, labelpad=10, fontweight='bold')
    ax.set_ylabel('Final Marks', fontsize=12, labelpad=10, fontweight='bold')
    
    ax.grid(True, linestyle='--', alpha=0.5, color=COLORS['grid'])
    ax.set_axisbelow(True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    return fig


def create_avg_marks_by_category_chart(df: pd.DataFrame) -> plt.Figure:
    """
    Chart 4 — Average Marks by Attendance Category
    Bar chart comparing mean final marks across Excellent, Good, Average, and Poor groups.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    order = ['Excellent', 'Good', 'Average', 'Poor']
    colors = [COLORS['excellent'], COLORS['good'], COLORS['average'], COLORS['poor']]
    
    grouped = df.groupby('Attendance_Category')['Final_Marks'].mean()
    means = [grouped.get(cat, 0.0) for cat in order]
    
    bars = ax.bar(order, means, color=colors, width=0.52, edgecolor='#1E293B', linewidth=0.8)
    
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=11, fontweight='bold', color='#111827')
                        
    ax.set_title('Average Final Marks by Attendance Category', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Attendance Category', fontsize=12, labelpad=10, fontweight='bold')
    ax.set_ylabel('Mean Final Marks', fontsize=12, labelpad=10, fontweight='bold')
    
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color=COLORS['grid'])
    ax.set_axisbelow(True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    return fig


def create_top_students_attendance_chart(df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """
    Chart 5 — Top Students by Attendance
    Horizontal bar chart showing the highest attendance records.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    top_df = df.sort_values(by=['Attendance', 'Final_Marks'], ascending=False).head(top_n).copy()
    top_df = top_df.iloc[::-1]  # Reverse for top-down display in horizontal bar chart
    
    y_labels = [f"{row['Student_Name']} ({row['Student_ID']})" for _, row in top_df.iterrows()]
    attendances = top_df['Attendance'].values
    
    bar_colors = [COLORS['excellent'] if a >= 90 else COLORS['good'] for a in attendances]
    bars = ax.barh(y_labels, attendances, color=bar_colors, height=0.6, edgecolor='#1E293B', linewidth=0.7)
    
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width:.1f}%',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center',
                    fontsize=10, fontweight='bold', color='#111827')
                    
    ax.set_title(f'Top {min(top_n, len(df))} Students by Attendance Percentage', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Attendance (%)', fontsize=12, labelpad=10, fontweight='bold')
    ax.set_ylabel('Student', fontsize=12, labelpad=10, fontweight='bold')
    
    ax.set_xlim(0, 112)
    ax.grid(axis='x', linestyle='--', alpha=0.5, color=COLORS['grid'])
    ax.set_axisbelow(True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    return fig
