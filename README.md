# Attendify – Student Attendance Analysis

> **Analyze Attendance, Academic Performance and Student Trends**
> Academic Data Analysis Web Application for 2nd Year Computer Science & Engineering (CSE)

---

## 1. Project Overview

**Attendify** is a professional, beginner-friendly web application designed to analyze student attendance records, evaluate data quality, compute descriptive and inferential statistics, perform linear regression modeling, and render academic data visualizations.

The project demonstrates the practical application of:
- **NumPy**: High-performance numerical computations, statistical metrics, covariance and correlation matrices, and polynomial curve fitting (`np.polyfit`).
- **Pandas**: Tabular data manipulation, missing value detection (`isnull`), mean/mode imputation (`fillna`), duplicate detection and pruning (`drop_duplicates`), condition-based filtering, and data aggregation.
- **Matplotlib**: Generation of static, publication-ready academic plots (bar charts, scatter plots, trend lines, multi-category groupings).
- **Linear Regression**: Modeling the empirical relationship between attendance percentage (independent variable $X$) and final exam marks (dependent variable $Y$).
- **Streamlit**: Interactive, responsive Python web interface with dashboard metrics and dynamic forms.

---

## 2. Main Objectives

1. Maintain and manage student attendance datasets in CSV format.
2. Automate attendance calculation:
   $$\text{Attendance} = \left(\frac{\text{Classes Attended}}{\text{Total Classes}}\right) \times 100$$
3. Detect, audit, and clean missing, invalid, and duplicate student records.
4. Categorize student attendance into academic compliance tiers:
   - **Excellent**: $90\% - 100\%$
   - **Good**: $75\% - 89\%$
   - **Average**: $60\% - 74\%$
   - **Poor**: $< 60\%$
5. Monitor students falling below the mandatory $75\%$ attendance threshold.
6. Calculate Pearson correlation coefficient and linear regression parameters ($m$, $c$, $R^2$).
7. Predict final examination marks based on attendance percentages.
8. Deliver 5 publication-ready Matplotlib charts.
9. Dynamically generate actionable academic findings without hardcoded data.

---

## 3. Technology Stack

* **Programming Language:** Python 3.10+
* **Web Framework:** Streamlit
* **Data Processing & Manipulation:** Pandas
* **Numerical & Matrix Operations:** NumPy
* **Data Visualization:** Matplotlib
* **Data Storage:** Comma-Separated Values (CSV)

---

## 4. Dataset Schema (`attendance.csv`)

The dataset contains the following standardized columns:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `Student_ID` | String | Unique student roll / identification number (e.g. `S001`) |
| `Student_Name` | String | Full name of student |
| `Department` | String | Academic department (CSE, IT, ECE, MECH, etc.) |
| `Year` | Integer | Year of study ($1, 2, 3, 4$) |
| `Total_Classes` | Integer | Total lectures / lab sessions scheduled |
| `Classes_Attended` | Integer | Total lectures attended by student |
| `Attendance` | Float | Calculated attendance percentage ($0.0 - 100.0\%$) |
| `Study_Hours` | Float | Average independent study hours per week |
| `Internal_Marks` | Float | Internal continuous assessment marks ($0.0 - 100.0$) |
| `Final_Marks` | Float | Semester end final examination marks ($0.0 - 100.0$) |

---

## 5. Application Structure

```text
ATTENDANCE ANALYSICS/
│
├── app.py                      # Main Streamlit web application & page routing
├── attendance.csv              # Primary student dataset
├── requirements.txt            # Python dependencies
├── README.md                   # Complete academic documentation & viva guide
│
├── modules/
│   ├── __init__.py             # Module initialization
│   ├── data_processing.py      # Pandas cleaning, schema validation, NumPy stats
│   ├── regression.py           # NumPy linear regression, R², correlation & predictions
│   └── visualizations.py       # 5 professional Matplotlib figures
│
└── assets/
    └── logo.png                # Attendify brand icon
```

---

## 6. Page-by-Page Feature Walkthrough

### 1. Dashboard
- Displays core Key Performance Indicator (KPI) metrics: Total Students, Average Attendance, Average Final Marks, Students Below 75%, Highest Attendance, and Lowest Attendance.
- Summary table of student counts across attendance tiers.
- Quick distribution bar chart preview.
- Alert card highlighting students below the 75% attendance threshold.

### 2. Add Student
- Form with strict client-side validation (non-zero classes, marks between 0-100, no negative study hours, duplicate ID prevention).
- Computes attendance percentage automatically on the fly.
- Appends and saves the new record directly to `attendance.csv`.

### 3. Student Records
- Tabular record browser with column sorting.
- Interactive multi-select filters for Department and Academic Year.
- Range slider filter for attendance percentage.
- Real-time search by Student ID or Student Name.
- Soft-red highlighting of all student rows whose attendance is below 75%.
- Direct CSV export via `st.download_button`.

### 4. Data Cleaning
- Audits missing cells per column via `df.isnull().sum()`.
- Imputes missing numerical entries using column mean:
  ```python
  df[col] = df[col].fillna(df[col].mean())
  ```
- Imputes missing categorical entries using column mode.
- Detects duplicate roll numbers via `df.duplicated()` and purges them with `df.drop_duplicates()`.
- Interactive "Simulate Dirty Data" and "Clean Active Dataset Now" buttons for live viva presentations.

### 5. Attendance Analysis
- Deep-dive into student cohorts: Excellent (>=90%), Good (75-89%), Average (60-74%), and Poor (<60%).
- Prominently showcases core NumPy statistical functions:
  - `np.mean()`
  - `np.median()`
  - `np.std()`
  - `np.min()` / `np.max()`
  - `np.corrcoef()`
- Filtered tables for defaulters (<75%) and honor roll students (>=90%).

### 6. Regression Analysis
- Independent variable ($X$): Attendance (%).
- Dependent variable ($Y$): Final Marks.
- Calculates slope ($m$), intercept ($c$), correlation ($r$), and $R^2$ using NumPy `polyfit`.
- Displays the mathematical equation:
  $$\text{Final Marks} = m \cdot \text{Attendance} + c$$
- Interactive marks predictor tool: drag attendance slider to view projected final marks.
- Matplotlib scatter plot with the fitted regression line and 75% threshold marker.
- Scientific interpretation distinguishing correlation from causation.

### 7. Visualizations
5 dedicated Matplotlib charts rendered via tabs:
1. **Attendance Distribution**: Academic tier bar chart.
2. **Attendance vs Final Marks**: Scatter plot with regression fit line.
3. **Study Hours vs Final Marks**: Scatter plot with attendance colormap and trendline.
4. **Average Marks by Category**: Comparative bar chart of scores across tiers.
5. **Top 10 Students by Attendance**: Ranked horizontal bar chart.

### 8. Findings & Summary
- Dynamically synthesized report based on the active dataset.
- Lists cohort average, top student, lowest attendance student, defaulter count, correlation coefficient, and score impact per 10% attendance.
- Actionable recommendations for academic faculties and department heads.

---

## 7. Mathematical Formulation of Linear Regression

The linear model assumes that the relationship between Attendance ($X$) and Final Marks ($Y$) can be expressed as:

$$\hat{y}_i = m \cdot x_i + c$$

Where:
- $m$ is the slope:
  $$m = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}$$
- $c$ is the y-intercept:
  $$c = \bar{y} - m \cdot \bar{x}$$
- $r$ is the Pearson correlation coefficient:
  $$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \cdot \sum (y_i - \bar{y})^2}}$$
- $R^2$ (Coefficient of Determination) measures the proportion of variance in marks explained by attendance:
  $$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

In Attendify, these parameters are computed using NumPy's `np.polyfit(x, y, 1)` and `np.corrcoef(x, y)`.

---

## 8. Installation & Setup Instructions

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed.

### Step 1: Open Terminal in Project Directory
```bash
cd "d:\ATTENDANCE ANALYSICS"
```

### Step 2: Install Required Dependencies
```bash
python -m pip install -r requirements.txt
```

### Step 3: Run the Application

#### Option A: Run as a Web App in Browser
```bash
python -m streamlit run app.py
```
The browser will automatically open at `http://localhost:8501`.

#### Option B: Run as a Standalone Desktop Application (No Browser Bars)
* Simply double-click the **Attendify** icon on your **Windows Desktop**!
* Or run:
  ```cmd
  launch_app.bat
  ```
  or
  ```bash
  python desktop_app.py
  ```
  This launches Attendify in a dedicated, distraction-free desktop window with no URL bar, navigation buttons, or tabs — looking and feeling like native desktop software!

---

## 9. How to Turn This Web App Into a Mobile App (Android / iOS)

### Method 1: Progressive Web App (PWA) / Add to Home Screen (Recommended & Instant)
Streamlit apps are fully responsive and adapt cleanly to mobile screens.
1. Deploy the app for free to **[Streamlit Community Cloud](https://share.streamlit.io/)** (push repository to GitHub and click Deploy).
2. Open the URL in Google Chrome (on Android) or Safari (on iPhone).
3. Tap the browser menu (or Share button on iOS) and tap **"Add to Home Screen"**.
4. An **Attendify** app icon will appear on your smartphone's home screen. When tapped, it launches in full-screen standalone mode without any browser URL bars, looking exactly like an installed mobile app!

### Method 2: Wrap into an Android APK (WebView App)
If your college evaluation requires an actual `.apk` installer:
1. Create a lightweight Android Studio WebView project or use **Capacitor**:
   ```bash
   npm install @capacitor/core @capacitor/cli
   npx cap init Attendify com.attendify.app
   npx cap add android
   ```
2. Point the WebView URL to your hosted Attendify server URL.
3. Build the APK via `./gradlew assembleDebug` to install on any Android phone.

---

## 10. Viva Voce / Presentation Guide for 2nd Year CSE Students

When presenting this project to your professor or external examiner, follow this structured guide:

### Q1: Where and why is NumPy used in this project?
> **Answer:** NumPy is used for numerical operations and linear algebra. Specifically:
> - `np.mean()` & `np.median()`: Compute central tendencies of attendance and exam scores.
> - `np.std()`: Measure score and attendance dispersion.
> - `np.corrcoef()`: Generate the Pearson correlation matrix between attendance and final marks.
> - `np.polyfit(x, y, 1)`: Perform 1st-degree polynomial regression to compute the slope ($m$) and intercept ($c$).

### Q2: Where and why is Pandas used in this project?
> **Answer:** Pandas is used for dataframe handling, data cleaning, and aggregation:
> - `pd.read_csv()` & `to_csv()`: Persist and load dataset records.
> - `df.isnull().sum()`: Detect missing cells across each column.
> - `df.fillna(df.mean())`: Impute missing numeric values using column-wise means.
> - `df.duplicated()` & `df.drop_duplicates()`: Identify and remove duplicate student records.
> - `df.groupby('Attendance_Category')['Final_Marks'].mean()`: Group student cohorts to analyze comparative performance.

### Q3: Why is Matplotlib used instead of Plotly?
> **Answer:** Matplotlib is the fundamental plotting library in Python, standard in engineering and scientific research. It allows precise control over figure axes, ticks, annotations, and regression lines without external web dependencies.

### Q4: Does a high correlation prove that attending class causes higher marks?
> **Answer:** No. Correlation indicates association, not causation. While attending lectures provides direct instruction and teacher guidance, other confounding variables (such as hours of independent self-study, motivation, internal tests, and background knowledge) also influence marks.

### Q5: How does the Add Student page prevent bad data?
> **Answer:** It enforces boundary checks: classes attended cannot exceed total classes, total classes must exceed zero, marks must be constrained within $[0, 100]$, study hours cannot be negative, duplicate Student IDs are blocked, and attendance is calculated automatically to eliminate human error.

---

## 11. Future Enhancements

- **Multi-variable Multiple Linear Regression:** Incorporate `Study_Hours` and `Internal_Marks` simultaneously to predict `Final_Marks` ($Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3$).
- **Automated Email / SMS Notifications:** Alert students and proctors automatically when attendance dips below 75%.
- **Biometric Integration:** Connect directly to RFID or fingerprint attendance logs for automated daily sync.

---

## 12. License
This project is open-source and developed for academic educational purposes.
