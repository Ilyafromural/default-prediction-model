import datetime
import os
import argparse
import dill
import pandas as pd
import numpy as np
import time
import lightgbm
from preprocessing import *
from read_parquet import *
from normalisation import *

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(
    description='Choice of the method, data directory, target directory and model directory'
)
parser.add_argument(
    'method', type=str,
    choices=['fit', 'predict', 'metadata'],
    help='Choose a method to be run'
)
parser.add_argument(
    '-d', '--data_dir',
    required=False,
    default='./data/',
    help='Input data directory (default: ./data/)')
parser.add_argument(
    '-t', '--target_dir',
    required=False,
    default='./target/',
    help='Input target directory (default: ./target/)')
parser.add_argument(
    '-md', '--model_dir',
    required=False,
    default='./model/',
    help='Input directory for model to be saved in (default: ./model/)')
parser.add_argument(
    '-p', '--predict_dir',
    required=False,
    default='./data_for_predictions/',
    help='Input directory with data to make predictions (default: ./data_for_predictions/)')
args = parser.parse_args()


columns_to_read = [
    'id', 'rn', 'pre_since_opened', 'pre_since_confirmed', 'pre_pterm', 'pre_fterm',
    'pre_till_pclose', 'pre_till_fclose', 'pre_loans_credit_limit', 'pre_loans_next_pay_summ',
    'pre_loans_outstanding', 'pre_loans_total_overdue', 'pre_loans_max_overdue_sum',
    'pre_loans_credit_cost_rate', 'pre_loans5', 'pre_loans530', 'pre_loans3060', 'pre_loans6090',
    'pre_loans90', 'is_zero_loans5', 'is_zero_loans530', 'is_zero_loans3060', 'is_zero_loans6090',
    'is_zero_loans90', 'pre_util', 'pre_over2limit', 'pre_maxover2limit', 'is_zero_util',
    'is_zero_over2limit', 'is_zero_maxover2limit', 'enc_paym_0', 'enc_paym_1', 'enc_paym_2',
    'enc_paym_3', 'enc_paym_4', 'enc_paym_5', 'enc_paym_6', 'enc_paym_7', 'enc_paym_8',
    'enc_paym_9', 'enc_paym_10', 'enc_paym_11', 'enc_paym_12', 'enc_paym_13', 'enc_paym_14',
    'enc_paym_15', 'enc_paym_16', 'enc_paym_17', 'enc_paym_18', 'enc_paym_19', 'enc_paym_20',
    'enc_paym_21', 'enc_paym_22', 'enc_paym_23', 'enc_paym_24', 'enc_loans_account_holder_type',
    'enc_loans_credit_status', 'enc_loans_credit_type', 'enc_loans_account_cur', 'pclose_flag',
    'fclose_flag'
]


# Функция для загрузки данных, их преобразования и обучения модели
def fit():
    print('Default Prediction Pipeline. The fit method has been initiated')

    # Загружаем данные
    print("Loading data...")
    time.sleep(1)
    X = read_parquet_dataset_from_local(
        path_to_dataset=args.data_dir, start_from=0,
        num_parts_to_read=len(os.listdir(args.data_dir)),
        columns=columns_to_read, verbose=False
    )

    # Препроцессор для обработки данных
    preprocessor = Pipeline(steps=[
        ('preprocessing', FunctionTransformer(preprocessing))
    ])

    # Объявление модели для предсказания
    model = lightgbm.LGBMClassifier(
              seed=1,
              n_jobs=-1,
              is_unbalance='true',
              objective='binary',
              n_estimators=3000,
              learning_rate=0.05,
              max_depth=10,
              metric='auc',
              boosting_type='gbdt',
              num_leaves=62,
              verbose=-1)

    # Создаем итоговый пайплайн с моделью
    full_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Обучение препроцессора:
    print("Fitting data into preprocessor...")
    time.sleep(1)
    full_pipe.named_steps['preprocessor'].fit(X)

    # Трансформация данных при помощи препроцессора:
    print("Transforming data with preprocessor...")
    time.sleep(1)
    X_transformed = full_pipe.named_steps['preprocessor'].transform(X)

    # Загружаем целевую переменную
    print("Loading target...")
    time.sleep(1)
    y = pd.read_csv(
        filepath_or_buffer=(args.target_dir+sorted(os.listdir(args.target_dir))[-1]),
        index_col='id',
        nrows=len(X_transformed),
        dtype='int32'
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y,
        stratify=y['flag'],
        random_state=1,
        test_size=0.2
    )

    # Обучение модели:
    print("Training model...")
    time.sleep(1)
    full_pipe.named_steps['classifier'].fit(
        X_train, np.ravel(y_train),
        eval_set=[(X_test, np.ravel(y_test))],
        callbacks=[lightgbm.early_stopping(stopping_rounds=50)]
    )

    # Оценка модели по roc_auc
    score = round(dict(model.best_score_['valid_0'])['auc'], 4)
    # Печатаем метрику
    print(f'model: {type(model).__name__}\nroc_auc: {score}')

    # Запись модели и метаданных в файл
    print("Saving model in a file...")
    time.sleep(1)
    model_filename = args.model_dir + f'default_prediction_model_{datetime.datetime.now().strftime("%Y%m%d%H%M")}.pkl'
    with open(model_filename, 'wb') as file:
        dill.dump({
            'model': full_pipe,
            'metadata': {
                'name': 'Default prediction model',
                'author': 'Ilya Pachin',
                'version': 1,
                'date': f'{datetime.datetime.now()}',
                'type': type(model).__name__,
                'roc_auc': f'{score}'
            }
        }, file)

    print(f"Model saved as {model_filename}")


# Функция для предсказания дефолта по предоставленным данным
def predict():
    print('Default Prediction Pipeline. The predict method has been initiated')

    # Загрузка данных
    print("Loading data...")
    time.sleep(1)
    df = pd.read_parquet(
        path=(args.predict_dir+sorted(os.listdir(args.predict_dir))[-1]),
        columns=columns_to_read,
        engine='fastparquet'
    )

    # Считывание модели из pickle-файла
    print("Loading model...")
    time.sleep(1)
    model_filename = args.model_dir + sorted(os.listdir(args.model_dir))[-1]
    with open(model_filename, 'rb') as file:
        model = dill.load(file)

    # Преобразуем данные с помощью препроцессора
    print("Transforming data with preprocessor...")
    time.sleep(1)
    df = model['model'].named_steps['preprocessor'].transform(df)

    # Нормализуем полученный датафрейм для исключения появления ошибок
    print("Normalising data...")
    time.sleep(1)
    df = df_normalisation(df, model)

    # Выводим на печать предсказание модели
    print(f'Default prediction: {model["model"].named_steps["classifier"].predict(df)[-1]}')


# Функция, которая печатает метаданные модели
def metadata():
    print('Default Prediction Pipeline. The metadata method has been initiated')

    # Считывание модели из pickle-файла
    model_filename = args.model_dir + sorted(os.listdir(args.model_dir))[-1]
    with open(model_filename, 'rb') as file:
        model = dill.load(file)

    print(f'Model metadata:\n{model["metadata"]}')


def main():

    if args.method == 'fit':
        fit()
    elif args.method == 'predict':
        predict()
    else:
        metadata()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
