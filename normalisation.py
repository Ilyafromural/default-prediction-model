import pandas as pd


# Функция для нормализации (приведения к виду, необходимому для загрузки в модель машинного обучения)
def df_normalisation(df, model):
    # Список признаков (фичей), на которых обучалась модель
    feature_names_in_model = list(model['model'].named_steps['classifier'].feature_name_)
    # Список признаков (фичей) имеющегося датафрейма
    feature_names_in_df = list(df.columns)
    # Список недостающих признаков (фичей)
    add_feature_names = list(set(feature_names_in_model) - set(feature_names_in_df))
    # Добавление недостающих признаков (фичей) в датафрейм для предсказания
    # Это необходимо для того, чтобы не вылетала ошибка:
    # ValueError: The feature names should match those that were passed during fit
    # Также присваиваем всем новым фичам значение 0 т.к. они отсутствуют
    new_df = pd.DataFrame(columns=add_feature_names)
    df = pd.concat([df, new_df], axis=1)
    df = df.fillna(0)
    # Ставим все фичи в том же порядке, что и в модели для избежания появления ошибки
    df = df[feature_names_in_model]
    return df
