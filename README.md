# 🚢 Titanic: Machine Learning from Disaster (End-to-End Pipeline)

## 📌 Project Overview
This repository contains a professional, end-to-end Machine Learning pipeline for the classic Kaggle Titanic dataset. The objective is to predict passenger survival based on demographic and ticket information. 

Rather than relying on basic imputation and default algorithms, this project heavily emphasizes **Strategic Feature Engineering**, **Outlier Neutralization**, and **Hyperparameter Tuning** to achieve an accuracy score well beyond the standard baseline.

## 🧠 Hypothesis Generation & Domain Knowledge
Before touching the data, several core hypotheses were established:
1. **Socio-Economic Status (Pclass):** 1st-class passengers were placed higher on the ship and likely had priority for lifeboats. 
2. **Gender & Age (Sex, Age):** Historical maritime protocol dictated "Women and Children First," making these the strongest potential predictors.
3. **Family Cohesion (SibSp, Parch):** Families tend to stick together in emergencies. Surviving or perishing was likely linked to family size.

## 🛠️ Methodology & Feature Engineering
* **Smart Imputation:** Missing `Age` values were NOT filled using a global mean/median. Instead, passenger `Title` (Mr., Mrs., Master., etc.) was extracted from the `Name` string, and missing ages were imputed using the median age of their specific title group.
* **Feature Consolidation:** `SibSp` and `Parch` were combined into a singular `FamilySize` feature to prevent multicollinearity.
* **Outlier Neutralization (Binning):** `Age` was converted from continuous integers into categorical life stages (Child, Teen, Adult, Senior) to neutralize the mathematical pull of age outliers (e.g., 80-year-olds).
* **Magnitude Scaling:** The `Fare` feature was scaled using `StandardScaler` to prevent algorithms from falsely equating high monetary values with higher feature importance.
* **Unseen Category Handling:** Built a robust 'Rare' category to catch unseen variables in the test set (e.g., the title "Dona").

## ⚙️ Modeling & Hyperparameter Tuning
Four base models were tested using 5-Fold Cross-Validation to ensure robust scoring without Data Leakage:
* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors (KNN)

**Model Optimization:**
Using `GridSearchCV`, the top models were aggressively fine-tuned. 
* **Final Random Forest Score:** `83.28%`
* **Final Support Vector Machine (SVM) Score:** `83.28%`

## 📁 Repository Structure
* `/data` - Contains the train and test CSV files.
* `/notebooks` - The core Jupyter Notebook containing EDA, engineering, and tuning.
* '/results' - The submissions
* `GLOSSARIUM.md` - A master document of Data Science strategies, mental models, and lessons learned during this project.
* 'README.md' - Project Description