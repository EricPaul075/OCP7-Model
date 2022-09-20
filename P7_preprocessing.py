# **************************************************************************
# Fonctions de preprocessing du projet P7 importées de:
# https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features/script
# **************************************************************************

from P7_functions import *
import numpy as np
import pandas as pd
import gc # garbage collector

num_pattern_list = ['_AGE', 'AMT_', '_AVG', 'CNT_', 'DAYS_', '_MAX', '_MEAN', '_MEDI', '_MIN', '_MODE', 'NUM_', '_SUM', '_VAR']
cat_pattern_list = []

def get_var():
    """
    Renvoie les variables globales categorical_columns et numerical_columns
    en assurant une liste d'exemplaires uniques.
    :return: list, list, listes des features catégorielles et numériques.
    """
    return list(set(categorical_columns)), list(set(numerical_columns))


def application_train_test(num_rows=None, nan_as_category=False, rm_XNA=True):
    """
    Chargement et preprocessing des données de 'application_train.csv' et
        application_test.csv, avec création de nouvelle features
    :param num_rows: int, paramètre nrows de read_csv (nombre de lignes à lire)
    :param nan_as_category: bool, traitement des NaN comme une catégorie
    :param rm_XNA: bool, suppression des lignes 'CODE_GENDER'='XNA'
    :return: dataframe, données du fichier
    """
    global numerical_columns, categorical_columns

    # Chargement du fichier
    df = pd.read_csv(input_data_path + 'application_train.csv',
                     encoding='utf-8-sig',
                     encoding_errors='surrogateescape',
                     low_memory=False, nrows=num_rows)
    print(f"Taux de 'TARGET' à 1: {100.0*df['TARGET'].sum()/len(df[~df['TARGET'].isnull()]):.2f}%")
    test_df = pd.read_csv(input_data_path + 'application_test.csv',
                          encoding='utf-8-sig',
                          encoding_errors='surrogateescape',
                          low_memory=False, nrows=num_rows)
    df = pd.concat([df, test_df], ignore_index=True)

    # Optionnel: Supprime 4 lignes de demande de prêt avec CODE_GENDER=XNA
    if rm_XNA:
        df = df[df['CODE_GENDER'] != 'XNA']

    # Codage des variables catégorielles binaires
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])

    cat_columns = ['TARGET', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']

    # Features catégorielles avec valeur binaire (0, 1)
    other_bin_features = [f for f in df.columns.tolist()
                          if df[f].dtypes=='int64'
                          and f not in cat_columns
                          and set(pd.unique(df[f]))=={0,1}]
    cat_columns.extend(other_bin_features)

    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category=nan_as_category)
    cat_columns.extend(cat_cols)

    # Remplacement de DAYS_EMPLOYED=365.243 par NaN
    # Concerne les personnes retraitées (NAME_INCOME_TYPE=Pensioner)
    # mais 10 exceptions parmi les retraités
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    # Création de nouvelles features
    # Nombre de jours travaillés / Nombre de jours dans la vie du demandeur
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    # Revenu annuel / Montant du prêt
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    # Revenu annuel / Nombre de personnes de la famille
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    # Annuité du prêt / Revenu annuel
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    # Annuité du prêt / Montant du crédit
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    # Colonnes catégorielles et numérique
    categorical_columns = cat_columns.copy()
    num_columns = [nc for nc in df.columns.tolist() if nc not in cat_columns]
    numerical_columns = num_columns.copy()
    print("Nombre de colonnes catégorielles:", len(cat_columns))
    print("Nombre de colonnes numériques:", len(num_columns))

    del cat_columns, num_columns
    gc.collect()

    return df


def bureau_and_balance(num_rows=None, nan_as_category=True):
    """
    Preprocessing des 2 fichiers de données:
        → agrège bureau_balance autour de 'SK_ID_BUREAU'
        → fusionne les 2 fichiers
        → agrège bureau autour de 'SK_ID_CURR' de 3 manières:
            - globale (BURO_)
            - sur les crédits actifs seulement (ACTIVE_)
            - sur les crédits clos seulement (CLOSED_)
        → fusionne les 3 agrégations
    :param num_rows: int, paramètre nrows de read_csv (nombre de lignes à lire)
    :param nan_as_category: bool, traitement des NaN comme une catégorie
    :return: dataframe, données concernant les prêts précédents
        contractés auprès d'un autre organisme de crédit
    """
    global numerical_columns, categorical_columns
    
    bureau = pd.read_csv(input_data_path + 'bureau.csv', encoding='utf-8-sig',
                         encoding_errors='surrogateescape', low_memory=False,
                         nrows=num_rows)
    bb = pd.read_csv(input_data_path + 'bureau_balance.csv', encoding='utf-8-sig',
                     encoding_errors='surrogateescape', low_memory=False,
                     nrows=num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category=nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category=nan_as_category)

    # Agrégation et fusion de bureau_balance avec bureau
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']  # moyenne appliquée sur les nombres seulement
    bureau = agg_df_withdict(bureau, bb, bb_aggregations, 'SK_ID_BUREAU')

    # Dictionnaire d'agrégation pour les variables numériques
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']

    # Agrégation de bureau sur 'SK_ID_CURR' avec num_aggregations et cat_aggregations → features BURO_
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    # L'agrégation avec 'mean' transforme les variables catégorielles en numérique
    cat_columns = []
    #cat_columns = ['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist() if e[0] in cat_aggregations.keys()]
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

    # Agrégation de bureau pour les crédits actifs seulement → features ACTIVE_
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()

    # Agrégation de bureau pour les crédits fermés seulement → features CLOSED_
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')

    categorical_columns.extend(cat_columns.copy())
    num_columns = [nc for nc in bureau_agg.columns.tolist() if nc not in cat_columns]
    numerical_columns.extend(num_columns.copy())
    print("Nombre de features catégorielles:", len(cat_columns))
    print("Nombre de features numériques:", len(num_columns))

    del closed, closed_agg, bureau, cat_columns, num_columns
    gc.collect()
    return bureau_agg


# Preprocess previous_application.csv
def previous_application(num_rows=None, nan_as_category=True):
    """
    Preprocessing de 'previous_application':
        → agrège autour de 'SK_ID_CURR' de 3 manières:
            - globale (PREV_)
            - sur les crédits acceptés seulement (APPROVED_)
            - sur les crédits refusés seulement (REFUSED_)
        → fusionne les 3 agrégations
    :param num_rows: int, paramètre nrows de read_csv (nombre de lignes à lire)
    :param nan_as_category: bool, traitement des NaN comme une catégorie
    :return: dataframe, données concernant les prêts précédents
        contractés auprès du même organisme de crédit
    """
    global numerical_columns, categorical_columns
    
    prev = pd.read_csv(input_data_path + 'previous_application.csv', encoding='utf-8-sig',
                       encoding_errors='surrogateescape', low_memory=False, nrows=num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category=nan_as_category)

    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']

    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    # L'agrégation avec 'mean' transforme les variables catégorielles en numérique
    cat_columns = []
    #cat_columns = ['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist() if e[0] in cat_aggregations.keys()]
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')

    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')

    categorical_columns.extend(cat_columns.copy())
    num_columns = [nc for nc in prev_agg.columns.tolist() if nc not in cat_columns]
    numerical_columns.extend(num_columns.copy())
    print("Nombre de features catégorielles:", len(cat_columns))
    print("Nombre de features numériques:", len(num_columns))

    del refused, refused_agg, approved, approved_agg, prev, cat_columns, num_columns
    gc.collect()
    return prev_agg


def pos_cash(num_rows=None, nan_as_category=True):
    """
    Preprocessing de 'POS_CASH_balance':
        → agrège autour de 'SK_ID_CURR' (POS_)
        → crée la variable 'POS_COUNT': nombre de situations de cash sur
            les prêts précédemment demandés
    :param num_rows: int, paramètre nrows de read_csv (nombre de lignes à lire)
    :param nan_as_category: bool, traitement des NaN comme une catégorie
    :return: dataframe, données concernant le nombre de situations de cash sur
        les prêts précédemment demandés auprès du même organisme de crédit
    """
    global numerical_columns, categorical_columns
    
    pos = pd.read_csv(input_data_path + 'POS_CASH_balance.csv', encoding='utf-8-sig',
                         encoding_errors='surrogateescape', low_memory=False, nrows=num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=nan_as_category)

    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    # L'agrégation avec 'mean' transforme les variables catégorielles en numérique
    cat_columns = []
    #cat_columns = ['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist() if e[0] in cat_cols]
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])

    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()

    categorical_columns.extend(cat_columns.copy())
    num_columns = [nc for nc in pos_agg.columns.tolist() if nc not in cat_columns]
    numerical_columns.extend(num_columns.copy())
    print("Nombre de features catégorielles:", len(cat_columns))
    print("Nombre de features numériques:", len(num_columns))

    del pos, cat_columns, num_columns
    gc.collect()
    return pos_agg


def installments_payments(num_rows=None, nan_as_category=True):
    """
    Preprocessing de 'installments_payments':
        → crée des features (PAYMENT_PERC, PAYMENT_DIFF, DPD)
        → agrège autour de 'SK_ID_CURR' (INSTAL_)
        → crée la variable 'INSTAL_COUNT': nombre d'échéanciers
            de prêts précédemment demandés
    :param num_rows: int, paramètre nrows de read_csv (nombre de lignes à lire)
    :param nan_as_category: bool, traitement des NaN comme une catégorie
    :return: dataframe, données concernant le nombre d'échéanciers pour les prêts
        précédemment demandés auprès du même organisme de crédit
    """
    global numerical_columns, categorical_columns
    
    ins = pd.read_csv(input_data_path + 'installments_payments.csv', encoding='utf-8-sig',
                      encoding_errors='surrogateescape', low_memory=False, nrows=num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=nan_as_category)

    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']

    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    # L'agrégation avec 'mean' transforme les variables catégorielles en numérique
    cat_columns = []
    #cat_columns = ['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist() if e[0] in cat_cols]
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])

    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()

    categorical_columns.extend(cat_columns.copy())
    num_columns = [nc for nc in ins_agg.columns.tolist() if nc not in cat_columns]
    numerical_columns.extend(num_columns.copy())
    print("Nombre de features catégorielles:", len(cat_columns))
    print("Nombre de features numériques:", len(num_columns))

    del ins, cat_columns, num_columns
    gc.collect()
    return ins_agg


# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows=None, nan_as_category=True):
    """
    Preprocessing de 'credit_card_balance':
        → agrège autour de 'SK_ID_CURR' (CC_)
        → crée la variable 'CC_COUNT': nombre de crédits précédemment demandés
    :param num_rows: int, paramètre nrows de read_csv (nombre de lignes à lire)
    :param nan_as_category: bool, traitement des NaN comme une catégorie
    :return: dataframe, données concernant les crédits précédemment demandés
        auprès du même organisme de crédit
    """
    global numerical_columns, categorical_columns
    
    cc = pd.read_csv(input_data_path + 'credit_card_balance.csv', encoding='utf-8-sig',
                     encoding_errors='surrogateescape', low_memory=False, nrows=num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category=nan_as_category)

    # General aggregations
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    # L'agrégation avec les 5 opérations transforme les variables catégorielles en numérique
    cat_columns = []
    #cat_columns = ['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist() if e[0] in cat_cols]
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])

    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()

    categorical_columns.extend(cat_columns.copy())
    num_columns = [nc for nc in cc_agg.columns.tolist() if nc not in cat_columns]
    numerical_columns.extend(num_columns.copy())
    print("Nombre de features catégorielles:", len(cat_columns))
    print("Nombre de features numériques:", len(num_columns))

    del cc, cat_columns, num_columns
    gc.collect()
    return cc_agg
