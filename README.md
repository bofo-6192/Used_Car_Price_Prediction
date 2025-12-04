# Used_Car_Price_Prediction
Predicts used car selling prices using a full ML pipeline: data cleaning, EDA, preprocessing, Random Forest, hyperparameter tuning, and model diagnostics. Achieves R² = 0.97 with strong feature insights on performance, age, and usage factors.

## Project Overview
This project builds a complete end-to-end regression model to predict the selling price of used cars using the “Car details v3” dataset. The pipeline includes data cleaning, exploratory data analysis, feature engineering, preprocessing with transformers, hyperparameter tuning, performance evaluation, and business-driven feature interpretation.

## Key Steps
### 1. Data Understanding
- 8128 rows, 13 features  
- Target: `selling_price`  
- Missing values in mileage, engine, max_power, torque, and seats  
- Initial distribution shows strong right skew in price

### 2. Data Cleaning & Feature Engineering
- Removed non-numeric units (e.g., “bhp”, “kmpl”, “Nm”) using regex  
- Converted mileage, engine, max_power, and torque into numeric values  
- Dropped non-informative column: `name`  

### 3. Exploratory Data Analysis
- Univariate: distribution plots for numeric and categorical columns  
- Bivariate: correlations and boxplots vs selling price  
- Key correlations:  
  - max_power (0.75)  
  - torque (0.62)  
  - engine (0.46)  
  - year (0.41)  
  - km_driven (–0.23)  
- Automatic and diesel cars show higher median selling prices

### 4. Preprocessing
Implemented with `ColumnTransformer`:
- Median imputation for numeric features  
- Most frequent imputation + OneHotEncoding for categorical features  
- All missing values handled successfully  

### 5. Hyperparameter Tuning
Tuned with `RandomizedSearchCV`  
Best Parameters:
{'max_depth': 38, 'max_features': None, 'min_samples_leaf': 1,
'min_samples_split': 8, 'n_estimators': 108}

**Final Test Performance:**
- MAE: 70,838  
- RMSE: 142,771  
- R²: 0.97  

### 6. Model Diagnostics
- Predicted vs actual scatter plot  
- Residual distribution  
- Residuals vs prediction spread  
- Error increases slightly for very high-priced cars (expected due to data sparsity)

### 9. Feature Importance (Top Predictors)
1. max_power  
2. year  
3. torque  
4. km_driven  
5. mileage  
6. engine  
7. seller_type  
8. transmission  
9. seats  

### Business Insights
- Higher engine performance strongly increases price  
- Newer cars with lower mileage sell for more  
- Automatic and diesel vehicles have higher valuation  
- Seller type influences pricing differences  
- Seating capacity plays a minor role  

## Conclusion
This project demonstrates a full professional ML workflow—from data understanding to model tuning—that achieves excellent predictive performance. With an R² of 0.97, the model provides reliable price predictions and meaningful insights into market dynamics, making it valuable for dealerships, pricing engines, or resale platforms.

## How to Run
1. Upload `Car details v3.csv`  
2. Open `Used_Car_Price_Prediction.ipynb`  
3. Run all cells in order  
