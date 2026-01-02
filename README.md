<div align="center">

# Placement Prediction

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

### Student Placement Classification Pipeline

_Predict student placement outcomes using machine learning_

_powered by comprehensive EDA and predictive modeling_

---

[Features](#project-objective) •
[Getting Started](#getting-started) •
[Performance](#model-performance) •
[Architecture](#methodology)

</div>

---

## Project Objective

Using the Placement Dataset, predict whether a student will be placed or not with:

- **Exploratory Data Analysis** to understand trends, patterns, and factors influencing placements
- **Predictive modeling** to determine placement outcomes
- **Target accuracy**: > 60%

---

## Project Structure

```
Placements/
├── placement_prediction.ipynb      # Initial EDA and modeling
├── placement_prediction_v2.ipynb   # Enhanced version with cross-validation
├── Placement_BeginnerTask01.csv    # Dataset (10,000 samples)
└── README.md                       # Project documentation
```

---

## Dataset Overview

The dataset contains **10,000 student records** with 12 features covering academic performance, skills, and extracurricular activities.

### Feature Description

| Feature                     | Type        | Description                                                |
| --------------------------- | ----------- | ---------------------------------------------------------- |
| `StudentID`                 | Integer     | Unique identifier for each student                         |
| `CGPA`                      | Float       | Cumulative Grade Point Average (scale: 4.0-10.0)           |
| `Internships`               | Integer     | Number of internships completed                            |
| `Projects`                  | Integer     | Number of projects undertaken                              |
| `Workshops/Certifications`  | Integer     | Count of workshops attended or certifications earned       |
| `AptitudeTestScore`         | Integer     | Score in aptitude assessment (0-100)                       |
| `SoftSkillsRating`          | Float       | Rating of communication and interpersonal skills (1.0-5.0) |
| `ExtracurricularActivities` | Categorical | Participation in extracurriculars (Yes/No)                 |
| `PlacementTraining`         | Categorical | Attended placement training programs (Yes/No)              |
| `SSC_Marks`                 | Integer     | Secondary School Certificate percentage (0-100)            |
| `HSC_Marks`                 | Integer     | Higher Secondary Certificate percentage (0-100)            |
| `PlacementStatus`           | Categorical | **Target Variable** - Placed / NotPlaced                   |

---

## Methodology

### Phase 1: Data Loading and Initial Exploration

This phase establishes the foundation for analysis by examining the raw dataset:

- **Data Import**: Load CSV dataset using Pandas
- **Structure Analysis**: Examine shape (10,000 rows × 12 columns), data types, and column names
- **Missing Values Check**: Identify null values across all features using `isnull().sum()`
- **Duplicate Detection**: Find and handle duplicate records
- **Statistical Summary**: Generate descriptive statistics using `describe()` for numerical and categorical columns
- **Target Distribution**: Analyze class balance between Placed vs NotPlaced students

### Phase 2: Exploratory Data Analysis (EDA)

A comprehensive analysis of the dataset to uncover patterns, relationships, and insights:

#### 2.1 Univariate Analysis - Numerical Features

- **Histograms**: Distribution plots for CGPA, AptitudeTestScore, SSC_Marks, HSC_Marks, SoftSkillsRating
- **KDE Plots**: Kernel Density Estimation to visualize probability density of each numerical feature
- **Box Plots**: Identify outliers and understand the spread (IQR) of values for each feature
- **Descriptive Statistics**: Mean, median, standard deviation, min/max values, quartiles

#### 2.2 Univariate Analysis - Categorical Features

- **Count Plots**: Frequency distribution for ExtracurricularActivities, PlacementTraining, PlacementStatus
- **Pie Charts**: Proportional representation of categorical variable categories
- **Value Counts**: Exact counts and percentages for each category

#### 2.3 Bivariate Analysis - Feature vs Target

- **Grouped Bar Charts**: Compare feature distributions across Placed vs NotPlaced groups
- **Box Plots by Target**: Visualize how numerical features differ between placement outcomes
- **Cross-tabulation**: Contingency tables for categorical features vs PlacementStatus
- **Statistical Significance**: Identify which features show strong separation between classes

#### 2.4 Correlation Analysis

- **Correlation Matrix**: Compute Pearson correlation coefficients for all numerical features
- **Heatmap Visualization**: Color-coded correlation matrix using Seaborn
- **Feature Relationships**: Identify multicollinearity and strongly correlated feature pairs
- **Target Correlation**: Rank features by their correlation with placement outcome

### Phase 3: Feature Engineering

Prepare data for machine learning models:

- **Encoding Categorical Variables**: Convert `ExtracurricularActivities`, `PlacementTraining`, and `PlacementStatus` to numerical format using Label Encoding
- **Feature Scaling**: Apply StandardScaler or MinMaxScaler to normalize numerical features
- **Feature Selection**: Remove irrelevant features (StudentID) and select predictive features
- **Train-Test Split**: Divide data into 80% training and 20% testing sets with stratification

### Phase 4: Model Building and Evaluation

#### Version 1 (`placement_prediction.ipynb`)

- Basic model training with Random Forest Classifier
- Initial accuracy assessment on test data

#### Version 2 (`placement_prediction_v2.ipynb`)

- **Cross-Validation**: 5-fold stratified cross-validation for robust model evaluation
- **Multiple Models Comparison**: Evaluate various classification algorithms
- **Hyperparameter Considerations**: Default parameters with potential for tuning
- **Comprehensive Metrics**:
  - **Accuracy**: Overall correctness of predictions
  - **Precision**: Proportion of positive predictions that are correct
  - **Recall**: Proportion of actual positives correctly identified
  - **F1-Score**: Harmonic mean of precision and recall
  - **Confusion Matrix**: Detailed breakdown of TP, TN, FP, FN

---

## Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Usage

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('Placement_BeginnerTask01.csv')

# Encode categorical variables
le = LabelEncoder()
df['ExtracurricularActivities'] = le.fit_transform(df['ExtracurricularActivities'])
df['PlacementTraining'] = le.fit_transform(df['PlacementTraining'])
df['PlacementStatus'] = le.fit_transform(df['PlacementStatus'])

# Prepare features and target
X = df.drop(['StudentID', 'PlacementStatus'], axis=1)
y = df['PlacementStatus']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")
```

---

## Key Findings

After comprehensive EDA, the following factors were identified as **key predictors** of placement:

| Factor            | Impact                                                          |
| ----------------- | --------------------------------------------------------------- |
| CGPA              | Strong positive correlation with placement                      |
| AptitudeTestScore | Higher scores significantly increase placement chances          |
| Internships       | Students with internship experience have higher placement rates |
| PlacementTraining | Participation in training programs correlates with placement    |
| SSC/HSC Marks     | Academic history influences placement outcomes                  |
| Projects          | Practical project experience improves placement probability     |

---

## Tech Stack

- **Python 3.8+** - Programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib** - Static visualizations
- **Seaborn** - Statistical data visualization
- **Scikit-learn** - Machine learning algorithms and evaluation metrics

---

## Model Performance

| Metric    | Target | Status                  |
| --------- | ------ | ----------------------- |
| Accuracy  | > 60%  | Achieved (See notebook) |
| Precision | -      | Calculated in v2        |
| Recall    | -      | Calculated in v2        |
| F1-Score  | -      | Calculated in v2        |

---

## References

- [Scikit-learn Classification Guide](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Seaborn Visualization Tutorial](https://seaborn.pydata.org/tutorial.html)

---

## License

This project is for educational purposes.

_Built as part of GDSC AIML learning track_
