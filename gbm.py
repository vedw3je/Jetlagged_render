import joblib
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from model import *  # Assuming model-related imports and setup are here

def objective(trial):
    # Define the hyperparameters to tune
    param_grid = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.0001, 0.1),
        'max_depth': trial.suggest_int('max_depth', -1, 100),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0),
        'lambda_l1': trial.suggest_uniform('lambda_l1', 0, 2.0),
        'lambda_l2': trial.suggest_uniform('lambda_l2', 0, 2.0)
    }

    # Initialize the model with the hyperparameters
    model = lgb.LGBMRegressor(**param_grid, random_state=42)

    # Train the model with early stopping
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
             )

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate and return the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    return mse

# Initialize the Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Best parameters
print("Best Parameters:", study.best_params)

# Best model
best_params = study.best_params
best_model = lgb.LGBMRegressor(**best_params, random_state=42)
best_model.fit(X_train, y_train)

# Evaluate the best model
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error with Optimized LightGBM: {mse}")
print(f"R^2 Score with Optimized LightGBM: {r2}")

# Save the best model
model_filename = 'flight_delay_prediction_model_lightgbm_bayesian.pkl'
joblib.dump(best_model, model_filename)
print(f"Model saved as {model_filename}")

# Save the scaler if you used it
scaler_filename = 'scaleroptimize.pkl'
joblib.dump(scaler, scaler_filename)
print(f"Scaler saved as {scaler_filename}")
