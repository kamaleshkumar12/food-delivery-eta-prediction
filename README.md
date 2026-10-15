# Food Delivery ETA Prediction

A Machine Learning project that predicts food delivery time using delivery, distance, weather, and traffic-related features.

## Project Overview

This project explores machine learning techniques to estimate food delivery duration. Multiple regression models are evaluated and compared to identify a suitable prediction model.

## Models Evaluated

- Mean Baseline
- Linear Regression
- Random Forest Regressor
- LightGBM Regressor
- XGBoost Regressor
- CatBoost Regressor
- Model blending

## Validation Results

| Model | MAE (minutes) | RMSE (minutes) | R² |
|---|---:|---:|---:|
| LightGBM L2 | 3.021 | 3.797 | 0.835 |
| LightGBM L1 | 3.045 | 3.876 | 0.828 |
| XGBoost | 3.009 | 3.788 | 0.835 |
| CatBoost | 3.034 | 3.822 | 0.832 |
| Blended Model | 3.010 | 3.796 | 0.835 |
| Random Forest | 3.086 | 3.887 | 0.827 |
| Linear Regression | 4.608 | 5.814 | 0.612 |
| Mean Baseline | 7.601 | 9.359 | -0.005 |

These are validation results, not final test-set results.

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- LightGBM
- XGBoost
- CatBoost
- Google Colab / Jupyter Notebook

## Evaluation Metrics

Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R².

## Limitations

Prediction performance depends on training data quality and whether real delivery conditions are represented. Independent test-set evaluation is required before production use.

## Author

Machine Learning Portfolio Project
