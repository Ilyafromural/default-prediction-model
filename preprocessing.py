import pandas as pd

from sklearn.preprocessing import OneHotEncoder


# Список колонок для кодирования
columns_to_encode = [
    'rn', 'pre_since_opened', 'pre_since_confirmed', 'pre_pterm', 'pre_fterm', 'pre_till_pclose',
    'pre_till_fclose', 'pre_loans_credit_limit', 'pre_loans_next_pay_summ', 'pre_loans_outstanding',
    'pre_loans_total_overdue', 'pre_loans_max_overdue_sum', 'pre_loans_credit_cost_rate',
    'pre_loans5', 'pre_loans530', 'pre_loans3060', 'pre_loans6090', 'pre_loans90', 'pre_util',
    'pre_over2limit', 'pre_maxover2limit', 'enc_paym_0', 'enc_paym_1', 'enc_paym_2', 'enc_paym_3',
    'enc_paym_4', 'enc_paym_5', 'enc_paym_6', 'enc_paym_7', 'enc_paym_8', 'enc_paym_9',
    'enc_paym_10', 'enc_paym_11', 'enc_paym_12', 'enc_paym_13', 'enc_paym_14', 'enc_paym_15',
    'enc_paym_16', 'enc_paym_17', 'enc_paym_18', 'enc_paym_19', 'enc_paym_20', 'enc_paym_21',
    'enc_paym_22', 'enc_paym_23', 'enc_paym_24', 'enc_loans_account_holder_type',
    'enc_loans_credit_status', 'enc_loans_credit_type', 'enc_loans_account_cur'
]


def preprocessing(df):
    # Кодируем выбранные колонки
    ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore', dtype='int16')
    encoded_features = ohe.fit_transform(df[columns_to_encode])
    df_encoded_features = pd.DataFrame(encoded_features, columns=ohe.get_feature_names_out())
    # Конкатенируем и удаляем ненужные колонки
    df_encoded = pd.concat([df, df_encoded_features], axis=1).drop(columns_to_encode, axis=1)
    # Группируем с агрегацией по сумме
    df_processed = df_encoded.groupby(['id'], as_index=False).agg('sum')
    # Заполняем пропуски значением 0
    df_processed = df_processed.fillna(0)
    df_processed = df_processed.set_index('id')
    return df_processed.astype(dtype='int16')
