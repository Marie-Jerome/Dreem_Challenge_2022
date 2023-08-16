import numpy as np
import random as rd
import optuna
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

def datasets(records_list):
    rd.shuffle(records_list)
    return records_list[:6],records_list[6:]

def f1_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    err = 1-f1_score(y_true, np.argmax(y_pred, axis = 1), average="macro")
    return 'f1_err', err

def objective(trial, df_feats, feat_cols, records_list):
    training_record,val_records = datasets(records_list)
    df_train = df_feats[df_feats["record"].isin(training_record)]
    df_val = df_feats[df_feats["record"].isin(val_records)]

    X_train = np.array(df_train[feat_cols])
    y_train = np.array(df_train["target"])
    X_val = np.array(df_val[feat_cols])
    y_val = np.array(df_val["target"])
    eval_set = [(X_val, y_val)]
    
    params = {
        "iterations": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }

    model = CatBoostClassifier(**params, silent=False, 
                               task_type = "GPU", devices="0:1",
                               )
    model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=20, 
              verbose=False)
    pred = model.predict(X_val)

    f1score = f1_score(y_val, pred, average ='macro')
    return f1score