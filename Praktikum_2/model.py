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


def prepare_df(sessions):
    # Удаляем колонку device_model
    sessions = sessions.drop(columns=["device_model"])

    # Для всех Desktop устройств указываем desktop в качестве бренда
    sessions["device_brand"] = sessions["device_brand"].where(sessions['device_category'] != "desktop", 'desktop')

    # Для каждого бренда определяем основную OS и сохраняем словарь
    os_dict = sessions.groupby(["device_brand", "device_os"], as_index=False).size().sort_values(["device_brand", "size"], ascending=[True, False]) \
            .groupby(["device_brand"], as_index=False).first().drop(columns=["size"]).set_index('device_brand')['device_os'].to_dict()

    # Заполняем пустые значения в колонке device_os основной OS для бренда
    sessions['device_os'] = sessions['device_os'].fillna(sessions['device_brand'].map(os_dict))

    # Заполним случайными парами os + brand
    missing_idx = sessions[(sessions['device_brand'].isna()) | (sessions['device_os'].isna())].index
    values_to_refill = (
        sessions[(sessions['device_brand'].notna()) & (sessions['device_os'].notna())][["device_os", "device_brand"]]
        .sample(len(missing_idx), random_state=42)
        .reset_index(drop=True)
    )

    sessions.loc[missing_idx, ["device_os", "device_brand"]] = values_to_refill.values

    # Разделяем ширину и высоту экрана. Приводим к числовому типу
    sessions[["screen_width", "screen_height"]] = pd.DataFrame(sessions["device_screen_resolution"].str.split('x').tolist()) \
                                                    .apply(pd.to_numeric, errors='coerce').astype('Int64')
    # Находим площадь устройства для каждой сессии
    sessions["screen_area"] = sessions["screen_width"] * sessions["screen_height"]

    # Считаем коэффициент площади
    sessions["screen_area_coeff"] = round(sessions["screen_area"] / sessions.groupby(["device_category"])["screen_area"].transform('mean'), 2)

    # При обучении модели выяснилось, что есть несколько NaN значений. Присвоим им 1
    sessions["screen_area_coeff"] = sessions["screen_area_coeff"].fillna(1)

    # Удаляем больше не нужные колонки
    sessions = sessions.drop(columns=["device_screen_resolution", "screen_width", "screen_height", "screen_area"])

    # Отберём страны СНГ
    sng_countries = {
        'Armenia', 'Azerbaijan', 'Belarus', 'Kazakhstan', 'Kyrgyzstan',
        'Moldova', 'Tajikistan', 'Turkmenistan', 'Uzbekistan', 'Ukraine'
    }

    def map_country(country):
        if country == 'Russia':
            return 'Russia'
        elif country in sng_countries:
            return 'SNG'
        else:
            return 'Other'

    sessions["geo_country"] = sessions["geo_country"].apply(map_country)

    # Список городов миллионников
    millionaires = ["Yekaterinburg", "Krasnoyarsk", "Kazan", "Novosibirsk", "Nizhny Novgorod", "Ufa",
                    "Chelyabinsk", "Tula", "Voronez", "Rostov-on-Don", "Omsk", "Perm", "Samara", "Volgograd"]

    def map_city(city):
        if city in ('Moscow', 'Saint Petersburg'):
            return city
        elif city in millionaires:
            return 'Millonaries'
        else:
            return 'Other'
        
    sessions["geo_city"] = sessions["geo_city"].apply(map_city)

    sessions["is_mobile"] = sessions["device_category"].apply(lambda x: int(x != 'desktop'))
    # Удалим ненужную колонку
    sessions = sessions.drop(columns=["device_category"])
    for col in ["device_os", "device_brand", "device_browser"]:
        if col == 'device_brand': 
            # Для бренда получим 6 категорий, т.к. одно место занимает desktop
            top_values = sessions[col].value_counts().head(6).index
        else:
            top_values = sessions[col].value_counts().head(5).index
        sessions[col] = sessions[col].where(sessions[col].isin(top_values), 'Other')

    sessions['visit_month'] = pd.to_datetime(sessions['visit_date']).dt.month
    sessions['visit_hour'] = pd.to_datetime(sessions['visit_time'], format='%H:%M:%S').dt.hour
    sessions['visit_weekday'] = sessions['visit_date'].dt.dayofweek + 1
    sessions = sessions.drop(columns=["visit_date", "visit_time"])
    sessions = sessions.fillna('NaN')
    return sessions


def main():
    args = parse_args()
    if not args.train_df and not args.model_params:
        raise ValueError('Either dataset for training, or model parameters should be specified')
    
    if args.train_df is None and args.save_model:
        warnings.warn('Train datasets are not specified, no saving will be done')

    # Список категориальных переменных
    cat_features = ['utm_medium', 'utm_source', 'utm_campaign', 'utm_adcontent', 'device_os', 'device_brand', 'device_browser', 'geo_country', 'geo_city']

    # Список числовых переменных
    num_features = ['visit_number', 'is_mobile', 'screen_area_coeff', 'visit_month', 'visit_hour', 'visit_weekday']

    # Список всех нужных нам колонок
    all_features = cat_features + num_features

    if args.train_df:
        print('Reading train dataset....................')
        df = pd.read_csv(args.train_df)
        df = df.fillna('NaN')
        # Разделим на фичи и целевую переменную
        X = df.drop(columns=['target', 'session_id', 'client_id'])
        y = df['target']
        
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
            model.fit(train_pool, eval_set=val_pool, use_best_model=True)

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
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        print('Model fitted....................')

        print(f'Model score: {roc_auc_score(y_val, model.predict_proba(X_val[all_features])[:, 1])}')
        if args.save_model:
            model.save_model(args.save_model, format="cbm", export_parameters=None, pool=None)
            print(f'Model saved to {args.save_model}....................')
    
    if args.model_params:
        model = CatBoostClassifier()
        model.load_model(args.model_params, format='cbm')
        print('Model loaded....................')

    print('Reading prediction dataset....................')
    df = pd.read_csv(args.predict_df, parse_dates=["visit_date"])
    df = prepare_df(df)

    X = df
    predictions = model.predict(X[all_features])

    predictions_df = pd.DataFrame()
    predictions_df['session_id'] = df['session_id']
    predictions_df['prediction'] = predictions
    predictions_df.to_csv(args.prediction_path, index=False)
    print('Done....................')



if __name__ == '__main__':
    main()