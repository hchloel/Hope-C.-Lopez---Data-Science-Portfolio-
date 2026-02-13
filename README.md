# Hi, I’m Hope, and welcome to my Data Science portfolio! 
This repository showcases a collection of end‑to‑end projects I have completed during my M.S. in Data Science program. Each project reflects the full analytical workflow — from data cleaning and feature engineering to exploratory analysis, statistical modeling, machine learning, and interpretation.
The notebooks in this repo use real‑world, publicly available datasets and demonstrate practical, industry‑aligned data science skills. My goal is to highlight not only technical proficiency, but also clear reasoning, thoughtful methodology, and strong communication of insights.


## Project 1 — Austin Animal Shelter Outcomes Analysis
**Notebook**: austin-animal-shelter-outcomes-analysis.ipynb

**Techniques**: EDA, feature engineering, hypothesis testing, regression, classification, clustering

**Overview**: This project analyzes 7,200+ animal outcome records from the Austin Animal Center. It includes extensive data cleaning, transformation, and modeling to understand what factors influence shelter stay duration and outcomes.

**Key Steps**:
- Standardized missing values and removed test records
- Cleaned categorical fields (Type, Sex, Spayed/Neutered, Breed, Color)
- Created engineered features:
  - Age at intake/outcome
  - Life stage classification
  - Color grouping (89 raw colors → 8 families)
  - Log and sqrt transformations
  - Capped outliers
  - Built standardized and normalized datasets for modeling
- Exploratory Data Analysis:
  - Summary statistics for all variables
  - Missingness visualization
  - Distribution plots (raw vs. standardized vs. normalized)
  - Correlation, covariance, and Spearman rank matrices
- Outcome glossary based on ShelterBuddy definitions
- Hypothesis Testing
  - Welch’s t‑test: Spayed/neutered animals have significantly different shelter durations than intact animals (p < 0.0001).
- Regression Modeling
  Target: Log‑transformed shelter duration
  Predictors: Age at intake, Sex, Spayed/Neutered
  - R² ≈ 0.226
  - Sterilization status is the strongest predictor
  - Older animals tend to have shorter stays
  - Sex has no significant effect
  - Classification Modeling
  Target: Spayed/Neutered (binary)
  Model: Logistic Regression
  - Accuracy: 0.83
  - ROC‑AUC: 0.82
  - High recall for spayed animals
  - Lower recall for intact animals due to class imbalance
- Clustering
  Method: K‑Means (k=2)
  Features: Age at intake, Log shelter duration, Sex
  - Silhouette Score: 0.41
  - PCA visualization shows moderate separation

## Project 2 — Wine, Stars, and Planets Analysis
**Notebook**: wine-stars-planets-analysis.ipynb

**Techniques**: Clustering, linear regression, logistic regression, model evaluation

**Overview**: This notebook contains three independent modeling exercises, each using a different dataset and technique.

**Exercise 1 — Wine Quality Clustering**
Goal: Use K‑Means to cluster red and white wines based on chemical properties.
Steps:
- Loaded and cleaned red/white wine datasets
- Combined them and added a type column
- Built a pipeline with StandardScaler + KMeans
- Evaluated clustering with Fowlkes–Mallows Index
Results:
- FMI = 0.982 → near‑perfect separation
- Cluster centers reveal meaningful chemical differences
- K‑Means naturally distinguishes red vs. white wines

**Exercise 2 — Predicting Star Temperature**
Goal: Build a linear regression model to predict stellar temperature.
Steps:
- Loaded stars.csv and assessed missingness
- Kept only rows with complete numeric data (596 rows)
- Trained/test split (75/25)
- Built linear regression model
Results:
- R² = 0.8802
- RMSE = 257.94 K (~4.7% of mean temperature)
- Strong predictive performance
- Coefficients reveal meaningful astrophysical relationships
- Residuals mostly normal with mild skew

**Exercise 3 — Classifying Planetary Year Length**
Goal: Predict whether a planet’s orbital period is shorter than Earth’s.
Steps:
- Built logistic regression model
- Evaluated with accuracy, precision/recall/F1, ROC‑AUC, confusion matrix
Results:
- Accuracy = 0.977
- ROC‑AUC = 0.99
- Very few misclassifications
- Strong, balanced performance across both classes

## Project 3 — Housing Price Analysis in R
**Notebook**: r-housing-linear-regression.Rmd

**Techniques**: Data wrangling, categorical encoding, visualization, simple & multiple linear regression, residual diagnostics, ANOVA model comparison (R)

**Overview**: This project demonstrates my experience performing statistical modeling in R using RMarkdown. I analyze a housing dataset to understand how structural features and amenities influence sale price. The workflow includes data cleaning, exploratory visualization, simple and multiple linear regression, model diagnostics, and formal model comparison.

**Key Steps**:
- Verified dataset structure and confirmed no missing values
- Converted six binary categorical variables (mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea) from "yes"/"no" to 0/1
- Converted furnishingstatus into a factor to preserve categorical structure
- Explored distributions using histograms and boxplots
- Visualized relationships using a correlation matrix
- Built a simple linear regression model (price ~ area)
- Built a multiple regression model incorporating additional predictors
- Evaluated model assumptions using residual plots and Q‑Q plots
- Compared models using RMSE, MSE, R², adjusted R², and ANOVA


## Skills Demonstrated Across the Portfolio
**Data Wrangling**
- Handling missingness
- Standardizing categorical values
- Encoding binary and multi‑level categorical variables (Python & R)
- Outlier detection and treatment
- Feature engineering (age calculations, transformations, grouping)
- Dataset merging and cleaning across multiple formats
**Exploratory Data Analysis**
- Summary statistics and distribution analysis
- Correlation and covariance matrices
- Spearman rank correlations
- Missingness visualization
- Histograms, boxplots, scatterplots, and density plots
- Correlation heatmaps and matrix visualizations (Python & R)
**Statistical Modeling**
- Welch’s t‑test
- - Levene’s test
- Simple and multiple linear regression (Python & R)
- Regression diagnostics (residual plots, Q‑Q plots, heteroscedasticity checks)
- ANOVA model comparison
- Interpretation of coefficients, significance, and model assumptions
**Machine Learning**
- Linear regression
- Logistic regression
- K‑Means clustering
- PCA visualization
- Model evaluation: R², adjusted R², RMSE, MSE
- Classification metrics: precision, recall, F1 score, ROC‑AUC
- Clustering metrics: silhouette score, Fowlkes–Mallows Index
**Tools & Libraries**
- Python: Pandas, NumPy, Matplotlib, Seaborn, Scikit‑learn
- R: Base R, RMarkdown, corrplot, factor handling, model diagnostics
- Custom utilities (ml_utils)
- Jupyter Notebook & RMarkdown for reproducible analysis

## Summary
This portfolio demonstrates a full range of data science capabilities — from cleaning messy real‑world data to building and evaluating statistical and machine learning models in both Python and R. Each project is structured to show clear reasoning, strong methodology, and thoughtful interpretation.

**Thank you for taking the time to visit my portfolio! :)**
