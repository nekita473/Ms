import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
from model import prepare_df
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_params', help='Params of trained model to load', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Загрузка модели
    model = CatBoostClassifier()
    model.load_model(args.model_params)

    st.title('Команда 1. Учебная задача "Сберавтоподписка". Предсказание целевого визита по его параметрам')

    uploaded_file = st.file_uploader("Загрузите CSV файл с необработанными данными визитов", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=["visit_date"])
         # Проверяем, чтобы нужные нам колонки были в файле
        main_columns = [
            'session_id', 'visit_date', 'visit_time', 'visit_number', 
            'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent',
            'device_category', 'device_os', 'device_brand', 
            'device_model', 'device_screen_resolution', 'device_browser',
            'geo_country', 'geo_city'
        ]

        missing_columns = [col for col in main_columns if col not in df.columns]

        if missing_columns:
            st.error(f"В файле отсутствуют следующие обязательные колонки: {', '.join(missing_columns)}")
        else:
            st.success("Файл успешно загружен и проверен. Все необходимые колонки на месте")

        st.write("Входные данные:")
        st.write(df)

        df = prepare_df(df)
        all_features = ['utm_medium', 'utm_source', 'utm_campaign', 'utm_adcontent', 'device_os', 'device_brand',
                         'device_browser', 'geo_country', 'geo_city', 'visit_number', 'is_mobile', 'screen_area_coeff',
                           'visit_month', 'visit_hour', 'visit_weekday']
        
        y_pred = model.predict(df[all_features])
    
        result = pd.DataFrame({
            "session_id": df["session_id"],
            "target": y_pred
        })

        st.write("Предсказания для каждой сессии:")
        st.write(result)

if __name__ == '__main__':
    main()