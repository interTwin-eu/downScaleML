import optuna
from optuna.integration import LightGBMPruningCallback
import logging
import pathlib
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# module level logger
LOGGER = logging.getLogger(__name__)


# Define the objective function for Optuna optimization without batch processing
def objective(trial, data, target, test_data, test_target):
    # Suggested hyperparameters using Optuna
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 300),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.01, 100),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.01, 100),
        'verbose': -1,
        'early_stopping': 100,
        'keep_training_booster': True,
        'device': "cuda"
    }

    LogConfig.init_log(f"Cluster {cluster_id}: Starting trial {trial.number}")

    # Split the data into training and validation sets
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.2, random_state=42)

    dtrain = lgb.Dataset(train_x, label=train_y, free_raw_data=False)
    dvalid = lgb.Dataset(valid_x, label=valid_y, reference=dtrain)

    # Train the model with pruning callback
    pruning_callback = LightGBMPruningCallback(trial, 'rmse')
    model = lgb.train(
        params,
        train_set=dtrain,
        valid_sets=[dtrain, dvalid],
        valid_names=['train', 'valid']
    )

    # Predict on the test set
    preds = model.predict(test_data)

    # Compute RMSE for the test set
    rmse = mean_squared_error(test_target, preds, squared=False)

    LogConfig.init_log(f"Cluster {cluster_id}: Finished trial {trial.number} with RMSE: {rmse}")
    
    return rmse