# **************************************************************************
# Fonctions de machine learning du projet P7 importées de:
# https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features/script
# **************************************************************************
import numpy as np

from P7_functions import *

import lightgbm
#print(f'- Version de la librairie LightGBM : {lightgbm.__version__}\n')
from lightgbm import LGBMClassifier # gradient boosting framework that uses tree based learning algorithms
from lightgbm import early_stopping # Create a callback that activates early stopping

from sklearn.metrics import roc_auc_score, roc_curve, auc, fbeta_score
from sklearn.model_selection import KFold, StratifiedKFold

from warnings import simplefilter, filterwarnings, resetwarnings
import re
import time


# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm_auc(df, num_folds, stratified=False, max_feat=10, alpha=1.0, debug=False, df_fromfile=None):
    """
    Effectue la modélisation LightGBM des données
        avec la métrique 'auc',
        calcule et sauvegarde la prédiction de test,
        calcule l'importance des features.
    :param df: dataframe, données d'entrée issues du preprocessing.
    :param num_folds: int, nombre de plis à mettre en œuvre.
    :param stratified: boolean, option de plis stratifiés,
        default=False
    :param max_feat: int, nombre de features à afficher dans
        le graphique d'importance des features, default=10.
    :param alpha: float, rapport "taux TP / taux FP" souhaité
        defaut=1.0
    :param debug: boolean, option de debug,
        default=False
    :param df_fromfile: charge df depuis le fichier
        spéficié au lieu de le passer en paramètre.
        defaut=None, pas de chargement
    :return: dataframe["feature", "importance", "fold"]
    """
    start_time = time.time()

    # Découpage en jeu d'entrainement et de test
    if df_fromfile is not None:
        df = pd.read_csv(df_fromfile, sep=';', encoding='utf-8-sig',
                         encoding_errors='surrogateescape', low_memory=False)
    # Pour éviter "LightGBMError: Do not support special JSON characters in feature name"
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    del df
    gc.collect()

    # Affichage
    print(Fore.BLACK + Style.BRIGHT + Back.WHITE
          + f"LightGBM avec métrique AUC:\n"
          + Style.RESET_ALL)
    print(f'Version de la librairie LightGBM : {lightgbm.__version__}')
    print(f"Démarrage de LightGBM. Train shape: {train_df.shape}, test shape: {test_df.shape}")

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
        cv_model = 'StratifiedKFold'
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)
        cv_model = 'KFold'

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    simplefilter(action='ignore')
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            #num_threads=-1,
            #nthread=4,
            n_jobs=-1,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            #verbosity=-1, #silent=-1, verbose=-1,
            )
        clf.fit(train_x, train_y,
                eval_set=[(train_x, train_y), (valid_x, valid_y)], # Evaluation sur le fold complet (train+valid)
                eval_metric='auc',
                early_stopping_rounds=200,  # → replace with early_stopping callback function?
                #calbacks=[early_stopping(stopping_rounds=200, first_metric_only=True, verbose=False)],
                verbose=2147483647, # Supprime verbose avec int32 inf
                )
        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print(f" → Fold {n_fold + 1}: AUC={roc_auc_score(valid_y, oof_preds[valid_idx]):.5f}")
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    resetwarnings()
    print(f"Score total: AUC={roc_auc_score(train_df['TARGET'], oof_preds):.5f}")
    elapsed = time.time() - start_time
    print(f"LightGBM avec {cv_model} exécuté en {elapsed_format(elapsed)}\n")

    # Trace la courbe ROC et calcule le seuil optimal
    filename = fig_path + 'LightGBM - ROC curve'
    if debug: filename = filename + ' - Debug version.png'
    else: filename = filename + '.png'
    rec, spe, thr = display_ROC(
        train_df['TARGET'], oof_preds, "LightGBM, Kernel importé", alpha=alpha, save=filename)
    print(f"Optimisation pour α={alpha:.2f}: "
          f"recall={100*rec:.1f}%, specificity={100*spe:.1f}%, threshold={thr:.4f}")

    # Sauvegarde de la prédiction sur le jeu de test
    test_df['TARGET_PROB'] = sub_preds
    test_df['TARGET'] = [1 if x >= thr else 0 for x in sub_preds]
    print(f"% de prédictions de la classe 1 avec le seuil de {thr:.2f} optimisé sur la courbe ROC: "
          f"{100.0 * test_df['TARGET'].sum() / len(test_df):.2f}%")
    if debug:
        filename = data_path + "P7_kernel_LightGBM_auc_debug.csv"
    else:
        filename = data_path + "P7_kernel_LightGBM_auc.csv"
    test_df[['SK_ID_CURR', 'TARGET_PROB', 'TARGET']].to_csv(filename, sep=';', index=False)

    # Graphe d'importance des features
    print('\nImportance des features:')
    display_importances(feature_importance_df, max_feat=max_feat, debug=debug)
    return feature_importance_df


# LightGBM GBDT with KFold or Stratified KFold
def kfold_lightgbm_metric(df, num_folds, func_metric, stratified=False, max_feat=10, debug=False, df_fromfile=None):
    """
        Effectue la modélisation LightGBM des données
            avec une métrique spécifique,
            calcule et sauvegarde la prédiction de test,
            calcule l'importance des features.
        :param df: dataframe, données d'entrée issues du preprocessing.
        :param num_folds: int, nombre de plis à mettre en œuvre.
        :param func_metric: fonction de calcul de la métrique.
        :param stratified: boolean, option de plis stratifiés,
            default=False
        :param max_feat: int, nombre de features à afficher dans
        le graphique d'importance des features, default=10.
        :param debug: boolean, option de debug,
            default=False
        :param df_fromfile: charge df depuis le fichier
            spéficié au lieu de le passer en paramètre.
            defaut=None, pas de chargement
        :return: dataframe["feature", "importance", "fold"]
        """
    start_time = time.time()

    # Découpage en jeu d'entrainement et de test
    if df_fromfile is not None:
        df = pd.read_csv(df_fromfile, sep=';', encoding='utf-8-sig',
                         encoding_errors='surrogateescape', low_memory=False)
    # Pour éviter "LightGBMError: Do not support special JSON characters in feature name"
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    del df
    gc.collect()

    # Affichage
    print(Fore.BLACK + Style.BRIGHT + Back.WHITE
          + f"LightGBM avec métrique fbeta_score:\n"
          + Style.RESET_ALL)
    print(f'Version de la librairie LightGBM : {lightgbm.__version__}')
    print(f"Démarrage de LightGBM. Train shape: {train_df.shape}, test shape: {test_df.shape}")

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
        cv_model = 'StratifiedKFold'
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)
        cv_model = 'KFold'

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]

    simplefilter(action='ignore')
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            #num_threads=-1,
            #nthread=4,
            n_jobs=-1,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            #verbosity=-1, #silent=-1, verbose=-1,
            )
        clf.fit(train_x, train_y,
                eval_set=[(train_x, train_y), (valid_x, valid_y)], # Evaluation sur le fold complet (train+valid)
                eval_metric=func_metric,
                early_stopping_rounds=200,  # → replace with early_stopping callback function?
                #calbacks=[early_stopping(stopping_rounds=200, first_metric_only=True, verbose=False)],
                verbose=2147483647, # Supprime verbose avec int32 inf
                )
        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print(f" → Fold {n_fold+1}: fbeta_score={func_metric(valid_y, oof_preds[valid_idx])[1]:.5f}")
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    resetwarnings()
    print(f"Score total: fbeta_score={func_metric(train_df['TARGET'], oof_preds)[1]:.5f}")
    elapsed = time.time() - start_time
    print(f"LightGBM avec {cv_model} exécuté en {elapsed_format(elapsed)}\n")

    # Seuil pour la classe 1
    thr = 0.5

    # Sauvegarde de la prédiction sur le jeu de test
    test_df['TARGET_PROB'] = sub_preds
    test_df['TARGET'] = [1 if x >= thr else 0 for x in sub_preds]
    print(f"% de prédictions de la classe 1 avec le seuil de {thr:.2f} : "
          f"{100.0 * test_df['TARGET'].sum() / len(test_df):.2f}%")
    if debug:
        filename = data_path + "P7_kernel_LightGBM_fbeta_debug.csv"
    else:
        filename = data_path + "P7_kernel_LightGBM_fbeta.csv"
    test_df[['SK_ID_CURR', 'TARGET_PROB', 'TARGET']].to_csv(filename, sep=';', index=False)

    # Graphe d'importance des features
    display_importances(feature_importance_df, max_feat=max_feat, debug=debug)
    return feature_importance_df


def predict_from_proba(predicted_proba, threshold):
    """
    Calcule la prédiction 0 ou 1, à partir des valeurs de probabilité et du seuil.
    :param predicted_proba: 1D ndarray, probabilités prédites par le modèle
    :param threshold: float, seuil au delà duquel la probabilité est 1.
    :return: 1D ndarray, prédictions binaires.
    """
    return [1 if value>=threshold else 0 for value in predicted_proba]


def display_ROC(label_true, label_pred, title, alpha=1.0, save=None):
    """
    Trace la courbe ROC à partir des labels, la droite y=α(1-x)
    et leur point d'intersection. Donne le seuil correspondant
    au point d'intersection.
    :param label_true: 1D ndarray, labels vrais.
    :param label_pred: 1D ndarray, labels prédits.
    :param title: str, titre à inclure dans celui du graphique.
    :param alpha: float, rapport "taux TP / taux FP" souhaité
    :return: float, float, float, recall, specificity, et
    threshold du point d'intersection.
    """
    fpr, tpr, thr = roc_curve(label_true, label_pred)
    line = alpha * (1 - fpr)

    # np.diff révèle les positions où le signe change (ie où les courbes se croisent)
    # np.argwhere donne les indices de ces positions où la valeur !=0
    idx = np.argwhere(np.diff(np.sign(tpr - line))).flatten()

    # Tracé des courbes
    plt.figure(figsize=(6, 6))
    plt.gca().fill_between(fpr, tpr, lw=2, color='steelblue', alpha=0.5)
    plt.plot(fpr, line, ls='solid', lw=2, color='coral', label=f"α={alpha:.2f} - thr={thr[idx][0]:.3f}")
    plt.plot(fpr[idx], line[idx], 'ro')
    plt.gca().set_ylim(bottom=0, top=1)
    plt.xlabel('FP rate', fontsize=12)
    plt.ylabel('TP rate', fontsize=12)
    title = "ROC - " + title + f" - AUC={auc(fpr, tpr):.2f}"
    plt.title(title, fontsize=14)
    plt.legend()
    plt.tight_layout()
    if save is not None:
        plt.savefig(save, dpi=300)
    plt.show()

    # Valeurs à retourner
    recall = fpr[idx][0]
    specificity = 1 - line[idx][0]
    threshold = thr[idx][0]
    return recall, specificity, threshold


def display_importances(feature_importance_df_, max_feat=10, debug=False):
    """
    Trace le graphique d'importance  des features.
    :param feature_importance_df_: dataframe["feature", "importance", "fold"]
    :return: None
    """
    cols = feature_importance_df_[["feature", "importance"]].groupby(
        "feature").mean().sort_values(by="importance", ascending=False)[:max_feat].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure()
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (moyenne sur les folds)', fontsize=14)
    plt.tight_layout()
    if not debug:
        filename = fig_path + 'LightGBM - feature importances.png'
    else:
        filename = fig_path + 'LightGBM - feature importances - Debug version.png'
    plt.savefig(filename, dpi=300)
    plt.show()



