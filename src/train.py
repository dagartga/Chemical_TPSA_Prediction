import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import config
import argparse
import model_dispatcher


def scale_data(df):

        to_scale = df.drop(columns=['kfold','toxic', 'FormalCharge'])
        scaler = StandardScaler()
        scaled_values  = scaler.fit_transform(to_scale)
        scaled_df = pd.DataFrame(scaled_values, columns = to_scale.columns)
        final_df = df[['kfold', 'toxic', 'FormalCharge']].join(scaled_df)

        return final_df


def run(fold, model, label):
    df = pd.read_csv(config.TRAINING_FILE)
    
    scaled_df = scale_data(df)

    df_train = scaled_df[scaled_df['kfold'] != fold].reset_index(drop=True)

    df_valid = scaled_df[scaled_df['kfold'] == fold].reset_index(drop=True)

    X_train = df_train.drop(label, axis=1).values
    y_train = df_train[label].values

    X_valid = df_valid.drop(label, axis=1).values
    y_valid = df_valid[label].values

    reg_model = model_dispatcher.models[model]
    reg_model.fit(X_train, y_train)

    y_preds = reg_model.predict(X_valid)

    mse = metrics.mean_squared_error(y_valid, y_preds)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_valid, y_preds)
    print(f'Fold: {fold}, RMSE:{round(rmse,4)}, R-Squared:{round(r2,4)}')

    joblib.dump(reg_model, os.path.join(config.MODEL_OUTPUT, f"{model}_{fold}.bin"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int)
    parser.add_argument("--model", type=str)
    parser.add_argument("--label", type=str)

    args = parser.parse_args()

    run(fold=args.fold,
        model=args.model,
        label=args.label)
