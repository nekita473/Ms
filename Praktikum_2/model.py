import argparse
import pandas as pd
import warnings
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_df', help='Train dataset', type=str)
    parser.add_argument('--save_model', help='Path to save model parameters', type=str)
    parser.add_argument('--model_params', help='Params of trained model to load', type=str)
    parser.add_argument('--predict_df', help='Dataset for prediction', type=str, required=True)
    parser.add_argument('--prediction_path', help='Path to save predictions', type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if not args.train_df and not args.model_params:
        raise ValueError('Either dataset for training, or model parameters should be specified')
    
    if args.train_df is None and args.save_model:
        warnings.warn('Train datasets are not specified, no saving will be done')
    


    if args.train_df:
        print('Reading train dataset....................')
        df = pd.read_csv(args.train_df)
        df = df.fillna('NaN')
        # Разделим на фичи и целевую переменную
        X = df.drop(columns=['target', 'session_id', 'client_id'])
        y = df['target']

        # Список категориальных переменных
        cat_features = ['utm_medium', 'utm_source', 'utm_campaign', 'utm_adcontent', 'device_os', 'device_brand', 'device_browser', 'geo_country', 'geo_city']

        # Список числовых переменных
        num_features = ['visit_number', 'is_mobile', 'screen_area_coeff', 'visit_month', 'visit_hour', 'visit_weekday']

        # Список всех нужных нам колонок
        all_features = cat_features + num_features

        print('Starting cross validation....................')
        scores = []
        skf = StratifiedKFold(n_splits=2)
        for train_index, val_index in skf.split(X, y):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            # Задаём параметры модели
            model = CatBoostClassifier(
                iterations       = 3000,
                learning_rate    = 0.05,
                depth            = 6,         # аналог num_leaves
                l2_leaf_reg      = 3,
                eval_metric      = 'AUC',
                loss_function    = 'Logloss',
                random_state     = 42,
                verbose          = 200,
                early_stopping_rounds = 200,
                class_weights    = {0: 1, 1: (y_train==0).sum() / (y_train==1).sum()},
                task_type        = 'GPU'
            )

            train_pool = Pool(data=X_train[all_features], label=y_train, cat_features=cat_features)
            val_pool   = Pool(data=X_val[all_features], label=y_val, cat_features=cat_features)

            # Обучаем
            model.fit(train_pool, eval_set = val_pool, use_best_model = True)

            # Считаем метрики
            y_pred = model.predict_proba(X_val[all_features])[:, 1]
            scores.append(roc_auc_score(y_val, y_pred))

        print(f'Cross val ROC-AUC score: {np.mean(scores)}')

        # Разделим на train и validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        train_pool = Pool(data=X_train[all_features], label=y_train, cat_features=cat_features)
        val_pool   = Pool(data=X_val[all_features], label=y_val, cat_features=cat_features)

        print('Fitting model....................')
        # model.fit(train_pool, eval_set = val_pool, use_best_model = True)
        model.load(args.model_params, format='cbm')
        print('Model fitted....................')

        print(f'Model score: {roc_auc_score(y_val, model.predict_proba(X_val[all_features])[:, 1])}')
        if args.save_model:
            model.save_model(args.save_model, format="cbm", export_parameters=None, pool=None)
            print(f'Model saved to {args.save_model}....................')
    
    if args.model_params:
        model.load(args.model_params, format='cbm')
        print('Model loaded....................')

    print('Reading prediction dataset....................')
    df = pd.read_csv(args.predict_df)
    df = df.fillna('NaN')

    df = df.drop(columns=['target', 'client_id']) # В тестовом датасете не будет колонки predict, но пока мы используем train, дропаем её

    X = df.drop(['session_id'], axis=1)
    predictions = model.predict(X)

    predictions_df = pd.DataFrame()
    predictions_df['session_id'] = df['session_id']
    predictions_df['prediction'] = predictions
    predictions_df.to_csv(args.prediction_path, index=False)
    print('Done....................')



if __name__ == '__main__':
    main()