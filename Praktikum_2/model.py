from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import argparse
import pandas as pd
import warnings
import pickle


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
    
    model = LogisticRegression()
    if args.train_df:
        print('Reading train dataset....................')
        df = pd.read_csv(args.train_df)
        X = df.drop(['target'], axis=1)
        y = df['target']

        print('Starting cross validation....................')
        cv = cross_validate(model, X, y, cv=5, scoring='roc_auc', verbose=10)
        print(f'Cross val ROC-AUC score: {float(cv["test_score"].mean())}')

        print('Fitting model....................')
        model.fit(X, y)
        print('Model fitted....................')
        if args.save_model:
            with open(args.save_model, 'wb') as file:
                pickle.dump(model, file)
            print(f'Model saved to {args.save_model}....................')
    
    if args.model_params:
        with open(args.model_params, 'rb') as file:
            model = pickle.load(file)
            print('Model loaded....................')

    print('Reading prediction dataset....................')
    df = pd.read_csv(args.predict_df)

    df = df.drop(['target'], axis=1) # В тестовом датасете не будет колонки predict, но пока мы используем train, дропаем её
    df['session_id'] = list(range(len(df))) # В тестовом датасете предполагается колонка session_id, но пока мы используем train, добавим её
    # Также тестовый датасет нужно обработать (выкинуть фичи, сохранённым OneHotEncoding обработать категориальные). Пока не сделано, но помним об этом

    X = df.drop(['session_id'], axis=1)
    predictions = model.predict(X)

    predictions_df = pd.DataFrame()
    predictions_df['session_id'] = df['session_id']
    predictions_df['prediction'] = predictions
    predictions_df.to_csv(args.prediction_path, index=False)
    print('Done....................')



if __name__ == '__main__':
    main()