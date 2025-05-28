# Учебная задача: Анализ сайта «СберАвтоподписка». Команда 1.

## Cross val ROC-AUC score: 0.728661560454394

Установить необходимые библиотеки (проверялось на Python 3.12):

```
pip install -r requirements.txt
```

Чтобы запустить модель, необходимо предоставить ей параметры:

--train_df - Предобработанный датасет для тренировки

--save_model - Куда сохранять параметры модели для дальнейшего использования

--model_params - Откуда загрузить параметры модели, если до этого её обучили и сохранили. Для примера есть файл best_model.cbm

--predict_df - Датасет для предсказаний (пока используется тот же, что и для тренировки). Обязательно

--prediction_path - Куда сохранить предсказания. Обязательно.

Пример запуска с предтренированной моделью:
```
python model.py --model_params best_model.cbm --predict_df test_sessions.csv --prediction_path test_preds.csv
```

Пример запуска с тренировкой:
```
python model.py --train_df data/df_prepared.csv --predict_df test_sessions.csv --prediction_path test_preds.csv --save_model best_model2.cbm
```
