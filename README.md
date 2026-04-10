# 🏡 Surprise Housing: Advanced Regression & Property Valuation

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-1.1.2-F7931E.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3.4-150458.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.20.3-013243.svg)](https://numpy.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.11.2-blueviolet.svg)](https://seaborn.pydata.org/)

> **Author:** Vinodh Nagarajaiah  
> **Programme:** AI/ML Executive Programme (UpGrad & IIIT-B)

## ⏱️ Executive Summary (TL;DR)
* **The Goal:** Build a robust, generalisable predictive model using Advanced Regression techniques (Ridge & Lasso) to accurately value prospective properties in the Australian housing market.
* **The Data:** Analysed a comprehensive dataset containing numerous categorical and numerical variables regarding property conditions, dimensions, and locations.
* **The Process:** Performed rigorous Exploratory Data Analysis (EDA), feature engineering, dummy variable creation, feature scaling, and implemented regularisation techniques to penalise model complexity.
* **The Result:** Successfully identified the most significant predictor variables for property prices and determined the optimal regularisation penalties ($\alpha$). Lasso Regression was ultimately selected as the optimal model due to its inherent feature selection capabilities.

---

## 📖 Table of Contents
1. [Business Problem & Objective](#-business-problem--objective)
2. [Skills & Machine Learning Competencies](#-skills--machine-learning-competencies)
3. [Methodology: The Modelling Pipeline](#-methodology-the-modelling-pipeline)
4. [Key Insights & Model Evaluation](#-key-insights--model-evaluation)
5. [Strategic Business Recommendations](#-strategic-business-recommendations)
6. [Future Scope & Improvements](#-future-scope--improvements)
7. [Repository Structure](#-repository-structure)
8. [Acknowledgements & Contact](#-acknowledgements--contact)

---

## 💼 Business Problem & Objective
**Surprise Housing**, a US-based real estate company, has decided to enter the Australian market. The company's business model relies on using data analytics to purchase houses at a price below their intrinsic value and subsequently flip them for a profit. 

**The Core Objective:** To model the price of houses using the available independent variables. The management requires a robust regression model incorporating **regularisation** to:
1. Identify which variables are statistically significant in predicting the price of a house.
2. Quantify how well those variables describe property prices.
3. Determine the optimal value of lambda ($\lambda$ / alpha) for both **Ridge** and **Lasso** regression to ensure the model does not overfit and generalises well to unseen market data.

---

## 🛠️ Skills & Machine Learning Competencies
* **Advanced Regression:** Implementation of Ridge (L2) and Lasso (L1) Regularisation to balance model accuracy and complexity.
* **Hyperparameter Tuning:** Finding the optimal penalty parameters ($\alpha$) to decrease model variance and prevent overfitting.
* **Feature Engineering:** Dummy variable creation for high-cardinality categorical data and treating multicollinearity.
* **Data Transformation:** Feature scaling to ensure uniform penalty application during regularisation.
* **Statistical Evaluation:** Assessing model robustness using $R^2$, Residual Sum of Squares (RSS), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

---

## 🧠 Methodology: The Modelling Pipeline

### 1. Data Cleaning & EDA
* Analysed the dataset to understand correlations between independent variables and the target variable (`SalePrice`).
* Handled missing values strategically to preserve dataset integrity.
* Conducted outlier treatment on the target variable to prevent skewing the regression coefficients.

### 2. Data Preparation
* Converted categorical variables into numeric formats using dummy encoding, significantly expanding the feature space.
* Split the data into Training and Testing sets to evaluate model generalisation.
* Applied feature scaling to independent variables. *(Crucial step: Regularisation heavily penalises large coefficients, making standardisation mandatory before applying Ridge or Lasso).*

### 3. Model Building & Regularisation
* **Ridge Regression (L2):** Applied Ridge to penalise large coefficients and reduce coefficient magnitude.
* **Lasso Regression (L1):** Applied Lasso to perform intrinsic feature selection by shrinking less important feature coefficients to exactly zero, resulting in a simpler, more interpretable model.

---

## 📊 Key Insights & Model Evaluation

### Optimal Alpha ($\alpha$) Values
* **Ridge Optimal Alpha:** `10.0`
* **Lasso Optimal Alpha:** `0.001`

### Model Performance ($R^2$ Scores)
* **Ridge Regression:** Train $R^2$: `0.94` | Test $R^2$: `0.93`
* **Lasso Regression:** Train $R^2$: `0.95` | Test $R^2$: `0.88`

*Note: Lasso was selected as the final model because the dataset contained a large number of variables, and reducing the feature space was a primary business goal.*

### Top Predictor Variables (Lasso Regression)
Based on the Lasso regression coefficients, the most significant variables driving Australian house prices include:
1. **OverallQual (8 & 9):** Overall material and finish quality.
2. **GrLivArea:** Above-ground living area square footage.
3. **Neighborhood_Crawfor:** Properties located in the Crawford neighbourhood.
4. **OverallCond (9):** Overall condition rating.
5. **Exterior1st_BrkFace:** Brick Face exterior coverings.

**Robustness Testing:** To ensure the model was generalisable, a secondary model was tested excluding the primary top 5 variables. The model adapted successfully, identifying `2ndFlrSF` (Second floor square feet), `Exterior1st_BrkFace`, `1stFlrSF` (First floor square feet), `TotalBsmtSF` (Total basement square feet), and `Functional_Typ` as the next strongest predictors.

---

## 💡 Strategic Business Recommendations
1. **Focus on Quality & Space:** The strongest positive predictors of property value are overwhelmingly related to the overall quality of materials (`OverallQual`) and total living area space (`GrLivArea`). Surprise Housing should prioritise undervalued homes that possess these core structural traits.
2. **Location is Quantifiable:** Certain neighbourhoods (like Crawford) command a significant premium. The company should overlay these geographical coefficients with current market listings to identify anomalies (houses priced below the model's geographic expectation).
3. **Deploying the Lasso Model:** Management should integrate the Lasso model into their purchasing pipeline. By automatically eliminating redundant variables, Lasso provides a clean, sparse equation that allows acquisition agents to quickly and reliably estimate a property's intrinsic value on the ground.

---

## 🚀 Future Scope & Improvements
While this project successfully establishes a robust baseline using regularised linear models, the predictive architecture can be advanced further:
1. **Non-Linear Ensemble Modelling:** Implement tree-based regressors such as **XGBoost** or **Random Forest** to capture complex, non-linear relationships in the housing data that Lasso/Ridge may miss.
2. **Advanced Feature Engineering:** Create derived temporal and spatial metrics, such as `HouseAge_At_Sale` or calculating geographic proximity to city centres and schools, rather than relying solely on categorical neighbourhood tags.
3. **Independent Variable Transformation:** Apply Box-Cox or Log transformations to highly right-skewed independent variables (e.g., `LotArea`) to further normalise the data space for linear algorithms.
4. **Interactive Deployment:** Package the optimal Lasso model into a **Streamlit** web application, allowing Surprise Housing's acquisition agents to input property specs and receive real-time, data-driven valuation estimates in the field.

---

## 📁 Repository Structure

    ├── ml2_advanced_regression_assignment.ipynb   # Main Jupyter Notebook (EDA, Modelling, Regularisation)
    ├── surprise_housing_au_train.csv              # Raw dataset used for analysis
    ├── surprise_housing_data_description.csv      # Metadata and definitions for dataset columns
    ├── surprise_housing_data_description.xlsx     # Excel format of data dictionary
    ├── subjective_question_ml2_adv.pdf            # Detailed answers to subjective assessment questions
    └── README.md                                  # Project overview and insights

---

## 🎓 Acknowledgments & Contact
This project is an assessment exercise designed and integrated into the AI/ML Programme at **UpGrad**, in collaboration with **IIIT-B**. 

**Created by:** Vinodh Nagarajaiah  

* 💼 **LinkedIn:** [vinodh-nagarajaiah](https://www.linkedin.com/in/vinodh-nagarajaiah/)
* 🐙 **GitHub:** [@techexorcist](https://github.com/techexorcist)
* ✉️ **Email:** [vinodh.nagarajaiah@gmail.com](mailto:vinodh.nagarajaiah@gmail.com)

<br>

> **Disclaimer:** *The dataset used in this project is for educational purposes only. All personally identifiable information (PII) has been removed or anonymised.*

---

## 📜 Licence
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT) - see the LICENSE file for details.
