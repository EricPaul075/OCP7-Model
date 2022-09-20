# *********************************************************************
# Fonctions associées au projet P7 - Implémentez un modèle de scoring *
# *********************************************************************
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
import gc # garbage collector
from contextlib import contextmanager # provides utilities for resource allocation to the 'with' statement

# Display options
from IPython.display import display, display_html, display_png, display_svg
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 199)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)

# Colorama
from colorama import init, Fore, Back, Style
#init()
# Fore: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
# Back: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
# Style: DIM, NORMAL, BRIGHT, RESET_ALL

# Répertoires
data_path = './P7_data/'
input_data_path = data_path + 'input_data/'
fig_path = './P7_fig/'


def elapsed_format(elapsed):
    """
    Formate le temps écoulé entre 2 time.time()
    :param elapsed: float, temps écoulé en secondes
    :return: str, durée formatée
    """
    h = int(elapsed / 3600)
    hh = '0' + str(h) if h<10 else str(h)
    m = int((elapsed - h * 3600) / 60)
    mm = '0' + str(m) if m < 10 else str(m)
    sec = elapsed - h * 3600 - m * 60
    s = int(sec)
    ss = '0' + str(s) if s < 10 else str(s)
    ms = int((sec - s) * 1000)
    if elapsed >= 60:
        return(f"{hh}:{mm}:{ss}")
    elif elapsed >= 1:
        return(f"{sec:.3f}s")
    else:
        return(f"{ms}ms")

import time
@contextmanager
def timer(process_title):
    """
    Mesure le temps d'exécution des instructions dans une section avec l'instruction 'with'.
    :param title: str, nom du processus dont on mesure le temps d'exécution
    :return: None
    """
    # Exécuté avant les instructions dans la section avec l'instruction 'with'
    start_time = time.time()
    # yield déclenche l'exécution des instructions dans la section avec l'instruction 'with'
    yield # équivalent à 'yield None'
    # Exécuté après l'exécution des instructions dans la section avec l'instruction 'with'
    elapsed = time.time() - start_time
    print(f"'{process_title}' exécuté en {elapsed_format(elapsed)}\n")


import glob
import os
def list_dir(dir_path, extension=None, verbose=False):
    """
    Liste les fichiers dans un répertoire, en se limitant
        de manière optionnelle à ceux de la spécification donnée
        par extension, ex: '*.csv'.
    :param dir_path: str, chemin du répertoire.
    :param extension: str, extension des fichiers à lister
        default: None, liste tous les fichiers.
    :param verbose: bool, mode verbose.
    :return: list, liste des fichiers
    """
    if extension is not None:
        path = dir_path + extension
    else:
        path = dir_path
    list_filenames = glob.glob(path)
    for index in range(len(list_filenames)):
        list_filenames[index] = list_filenames[index].replace('\\', '/')

    if verbose:
        print(f"Liste des {len(list_filenames)} fichiers de '{path}':")
        for file in list_filenames:
            filename = os.path.basename(file)
            print(" →", filename)

    return list_filenames


import csv
import os
def change_csv_sep(filepath, old_csv=',', new_csv=';', suffix=None):
    """
    Change le séparateur du fichier csv, notamment pour
        le rendre directement lisible par MS-Excel
    :param filepath: str, chemin complet du fichier
    :param old_csv: str, séparateur du fichier existant la place de l'existant
    :param suffix: str, pour créer un nouveau fichier identifié avec un suffixe
        default:None pour écrasement du fichier existant
    :return output_filepath: str, chemin du nouveau fichier (None si erreur)
    """
    if filepath[-4:] != '.csv':
        print("Error: filename must have '.csv' extension")
        output_filepath = None
    else:
        input_file = open(filepath, "r")
        reader = csv.reader(input_file, delimiter=old_csv)
        if suffix is not None:
            new_dir = os.path.dirname(filepath) + '/' + suffix + '/'
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            output_filepath = new_dir + os.path.basename(filepath)[:-4] + '_' + suffix + ".csv"
            print('output_filepath:', output_filepath)
        else:
            output_filepath = os.path.dirname(filepath) + "/tmp.txt"
        output_file = open(output_filepath, 'w', newline='')
        writer = csv.writer(output_file, delimiter=new_csv)
        writer.writerows(reader)
        input_file.close()
        output_file.close()
        if suffix is None:
            os.remove(filepath)
            os.rename(output_filepath, filepath)
            output_filepath = filepath
    return output_filepath


import os
def get_features_dict(list_filenames, verbose=False):
    """
    Construit le dictionnaire dont les clés sont les features
        et les valeurs les noms des fichiers dans lesquels on peut les trouver
    :param list_filenames: list, liste des fichiers de données
    :param verbose: bool, mode verbose
    :return: dict, dictionnaire des features
    """
    # Variables pour la fonction dataset_tables et get_data
    global features, n_features, data_dimensions, key_features
    n_features = {}
    features = {}
    data_dimensions = {}
    key_features = {}
    # Fonction get_features_dict
    features_dict = {}
    for filename in list_filenames:
        f_name = os.path.basename(filename)[:-4]
        if verbose: print(f"Lecture des features du fichier '{f_name}'")
        df = pd.read_csv(filename, encoding='utf-8-sig', encoding_errors='surrogateescape', low_memory=False)
        n_features[f_name] = df.shape[1]
        features[f_name] = df.columns.tolist()
        data_dimensions[f_name] = df.shape
        key_features[f_name] = []
        count = 0
        for feature in df.columns.tolist():
            if df.duplicated(subset=feature).any()==False:
                key_features[f_name].append(feature)
            if features_dict.get(feature)==None:
                count += 1
                features_dict[feature] = [f_name]
            else:
                features_dict[feature].append(f_name)
        if verbose: print(f" → {count} features sur {df.shape[1]} ajoutées au dictionnaire")
    return features_dict


def sort_n_filter_features_dict(features_dict, verbose=False):
    """
    Trie les features par valeurs décroissantes du nombre de fichiers
        dans lesquels elles sont présentes.
    Filtre les entrées non communes à plusieurs fichiers.
    :param features_dict: dict, dictionnaire des features
        (format: {feature: [fichiers de données]})
    :param verbose: bool, mode verbose.
    :return: dict, dictionnaire des features trié et filtré.
    """
    # Tri descendant des features en fonction du nombre de fichiers dans lesquels elles sont présentes
    fns_features_dict = {k: v for k, v in sorted(features_dict.items(), key=lambda x: len(x[1]), reverse=True)}
    # Filtrage des features représentées dans un seul fichier de données
    size = len(features_dict)
    keys_to_rm = []
    for k in fns_features_dict.keys():
        if len(fns_features_dict[k])<=1:
            keys_to_rm.append(k)
    for k in keys_to_rm:
        fns_features_dict.pop(k, None)
    new_size = len(fns_features_dict)
    if verbose: print(f"\nNombre total de features: {size} → "
                      f"features communes à plusieurs fichiers: {new_size}")
    return fns_features_dict


def get_dataset_info():
    """
    Donne les dimensions (shape) des fichiers de données ainsi que
        les features clés (celles capables d'indexer le fichier).
        Utilise 2 variables globales de la fonction 'get_features_dict'.
    :return: dict, dictionnaire de format {filename: shape}
    :return: dict, dictionnaire de format {filename: [features]}
    """
    return data_dimensions, key_features


def dataset_tables(filenames, features_dict, verbose=False):
    """
    Etablit les tables des relations entre les fichiers des jeux de données.
    :param filenames: list, liste des noms de fichier du dataset sans le chemin ni l'extension
    :param features_dict: dict, dictionnaire des features établi par get_features_dict ou sort_n_filter_features_dict
    :param verbose: bool, mode verbose
    :return: dataframe, dataframe, dataframe, dataframe
        - df_nrel, dataframe du nombre de features communes par paire de fichiers
        - df_feat, dataframe de la liste des features communes par paire de fichiers
        - df_keynrel, dataframe du nombre de features-clés communes par paire de fichiers
        - df_keyfeat, dataframe de la liste des features-clés communes par paire de fichiers
    """
    # Initialisation des tables
    df_nrel = pd.DataFrame(np.zeros((len(filenames), len(filenames)), dtype=int), index=filenames, columns=filenames)
    df_keynrel = pd.DataFrame(np.zeros((len(filenames), len(filenames)), dtype=int), index=filenames, columns=filenames)
    df_feat = pd.DataFrame([], index=filenames, columns=filenames)
    df_keyfeat = pd.DataFrame([], index=filenames, columns=filenames)
    for a in filenames:
        for b in filenames:
            if a==b:
                # Les valeurs sont produites par la fonction get_features_dict
                df_nrel.at[a, b] = n_features[a]
                df_feat.at[a, b] = features[a]
                df_keyfeat.at[a, b] = key_features[a]
                df_keynrel.at[a, b] = len(key_features[a])
            else:
                df_feat.at[a, b] = []
                df_keyfeat.at[a, b] = []

    # Constitution des tables df_nrel et df_feat
    for dict_key, dict_value in features_dict.items():
        pairs = [(a,b) for idx, a in enumerate(dict_value) for b in dict_value[idx+1:]]
        for pair in pairs:
            a, b = pair[0], pair[1]
            df_nrel.at[a,b] += 1
            df_nrel.at[b,a] += 1
            df_feat.at[a,b].append(dict_key)
            df_feat.at[b,a].append(dict_key)

    # Constitution de la table df_keyfeat
    for dict_key, dict_value in features_dict.items():
        pairs = [(a, b) for idx, a in enumerate(dict_value) for b in dict_value[idx + 1:]]
        for pair in pairs:
            a, b = pair[0], pair[1]
            kfs = list(set(key_features[a] + key_features[b]))
            if kfs :
                for kf in kfs:
                    if kf in df_feat.at[a,b] and kf not in df_keyfeat.at[a, b]:
                        df_keyfeat.at[a, b].append(kf)
                        df_keyfeat.at[b, a].append(kf)
                        df_keynrel.at[a, b] = len(df_keyfeat.at[a, b])
                        df_keynrel.at[b, a] = len(df_keyfeat.at[b, a])

    # Affichage des tables
    if verbose:
        print(Fore.BLACK + Style.BRIGHT + Back.WHITE
              + "Table des nombres de relations entre les fichiers du jeu de données:\n"
              + Style.RESET_ALL)
        display(df_nrel)
        print(Fore.BLACK + Style.BRIGHT + Back.WHITE
              + "Table des features mettant en relation les fichiers du jeu de données:\n"
              + Style.RESET_ALL)
        display(df_feat)
        print(Fore.BLACK + Style.BRIGHT + Back.WHITE
              + "Table des nombres de relations clés entre les fichiers du jeu de données:\n"
              + Style.RESET_ALL)
        display(df_keynrel)
        print(Fore.BLACK + Style.BRIGHT + Back.WHITE
              + "Table des features-clés mettant en relation les fichiers du jeu de données:\n"
              + Style.RESET_ALL)
        display(df_keyfeat)

    return df_nrel, df_feat, df_keynrel, df_keyfeat


import networkx as nx
print(f'- Version de la librairie networkx : {nx.__version__}')
def dataset_graph(df_nrel, df_feat, df_keynrel, df_keyfeat, max_eli=1, save=None,
                  with_labels=True, node_size=20000, node_shape='o', alpha=0.85):
    """
    Trace le graphe des relations (feature de même nom) entre les fichiers de données.
        Les relations contenant une feature clés sont labellisées.
        L'épaisseur des relations est proportionnelle au nombre de features communes.
    :param df_nrel: dataframe du nombre de relations entre paires de fichiers de données
    :param df_feat: dataframe des features entre paires de fichiers de données
    :param df_keynrel: dataframe du nombre de relations-clés (features clés) entre paires de fichiers de données
    :param df_keyfeat: dataframe des features-clés entre paires de fichiers de données
    :param max_eli: int, nombre max de features représentées dans la relation entre 2 fichiers
    :param save: str, nom du fichier (.png) de sauvegarde graphique
    :return: None
    """
    G = nx.Graph()

    # Création des nœuds
    nodes = df_nrel.columns.tolist()
    G.add_nodes_from(nodes)
    pos = nx.spring_layout(G)

    # Création des relations
    rel_edges = [(a,b,df_nrel.at[a,b])
             for idx, a in enumerate(nodes)
             for b in nodes[idx+1:]
             if df_nrel.at[a,b]>0 and df_keynrel.at[a,b]==0]
    G.add_weighted_edges_from(rel_edges, color='dimgray')

    key_edges = [(a,b, df_nrel.at[a,b])
                 for idx, a in enumerate(nodes)
                 for b in nodes[idx+1:]
                 if df_keynrel.at[a,b]>0]
    G.add_weighted_edges_from(key_edges, color='coral')

    edges = G.edges()
    colors = [G[u][v]['color'] for u, v in edges]
    weights = [G[u][v]['weight'] for u, v in edges]

    # Ajout des labels des relations impliquant une feature clé
    edge_labels = {(a, b): df_keyfeat.at[a, b][:max_eli]
                   for idx, a in enumerate(nodes)
                   for b in nodes[idx + 1:]
                   if df_keynrel.at[a,b]>0}

    # Tracé du graphe
    plt.figure(figsize=(15, 12))
    ax = plt.gca()
    ax.margins(0.08)
    nx.draw(G, pos, edge_color=colors, width=weights,
            with_labels=False, node_size=node_size,
            node_shape=node_shape, alpha=alpha)
    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=12,
                                font_color='k', font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.axis("off")
    plt.tight_layout()
    if save is not None:
        plt.savefig(fig_path+save, dpi=300)
    plt.show()
    return


def one_hot_encoder(df, nan_as_category=True):
    """
    Encode avec 'get_dummies' les colonnes de type 'object' d'un dataframe.
    :param df: dataframe, contient les colonnes à encoder.
    :param nan_as_category: bool, ajoute éventuellement une catégorie '_nan'.
        Cela permet en particulier d'imputer les NaN.
    :return: dataframe, list: le dataframe en entrée auquel est ajouté les colonnes encodées,
        et la liste des colonnes ajoutées
    """
    original_columns = df.columns.tolist()
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    del original_columns
    del categorical_columns
    gc.collect()
    return df, new_columns


def agg_df_withdict(df1, df2, agg_dict, key_feature, prefix='', drop_key=True, keep_df2=False):
    """
    Agrège 2 dataframes par la colonne 'key_feature'
        en regroupant les valeurs de df2
        selon le dictionnaire agg_dict
    :param df1: dataframe, avec feature-clé 'key_feature'
    :param df2: dataframe, à grouper par de 'key_feature' avec le dictionnaire agg_dict
    :param agg_dict: dict, du type {feature: list(operations)}
    :param key_feature: feature-clé de df1
    :param prefix: str, ajout d'un préfixe aux features
    :param drop_key: bool, supprime la colonne de la 'key_feature'
    :return: dataframe, résultant de l'agrégation des 2 dataframes
    """
    df2_agg = df2.groupby(key_feature).agg(agg_dict)
    # Concaténation des 2 niveaux de label de colonne (feature, opération)
    df2_agg.columns = pd.Index([str(prefix) + e[0] + "_" + e[1].upper() for e in df2_agg.columns.tolist()])
    df = df1.join(df2_agg, how='left', on=key_feature)
    if drop_key: df.drop([key_feature], axis=1, inplace=True)
    del df2_agg
    if not keep_df2: del df2
    gc.collect()
    return df
# Exemple
# ar1 = [('a', 1), ('b',2), ('c', 3)]
# ar2 = [('a', 1, 1), ('a', 3, 20), ('a', 11, 99), ('b', 4, 8), ('b', 0, -1), ('b', 20, 30), ('c', 9, 27)]
# df1 = pd.DataFrame(ar1, columns=['key', 'value1'])
# display(df1)
# df2 = pd.DataFrame(ar2, columns=['key', 'value2', 'value3'])
# display(df2)
# agg_dict1 = {'value2': 'mean'}
# agg_dict2 = {'value3': ['min', 'max']}
# agg_dict = {}
# for d in (agg_dict1, agg_dict2): agg_dict.update(d)
# print(agg_dict)
# prefix='AGG_'
# print(prefix + 'DF')
# display(agg_df_withdict(df1, df2, agg_dict, 'key', prefix=prefix))


def get_df_nan_rate(df, verbose=True):
    """
    Calcule le taux de NaN d'un dataframe.
    :param df: dataframe, dont on veut calculer le taux de NaN
    :param verbose: bool, mode verbose
    :return: float, taux de NaN (compris entre 0 et 1)
    """
    nan_rate = float(df.isnull().sum(axis=0).sum()) / (df.shape[0] * df.shape[1])
    if verbose: print(f"Taux de NaN: {100*nan_rate:.2f}%")
    return nan_rate

def check_for_inf(df, replace_with_nan=True, verbose=True):
    """
    Vérifie si le dataframe contient des valeurs infinie,
        affiche les informations en mode verbose, et
        remplace ces valeurs par np.NaN si replace_with_nan=True.
    :param df: dataframe, que l'on souhaite vérifier
    :param replace_with_nan: bool, remplace inf par np.NaN si True
        default=True
    :param verbose: bool, mode verbose pour afficher le nombre et
        taux de inf dans le dataframe et lister les features
        (colonnes) concernées.
    :return: dataframe modifié (sans affecter le dataframe en
        entrée) si replace_with_nan=True,
        int, nombre de valeurs infinies sinon.
    """
    nb_inf = 0
    col_with_inf = []
    for feature in df.columns:
        nb_inf_col = np.isinf(df[feature]).sum()
        if nb_inf_col > 0:
            nb_inf += nb_inf_col
            col_with_inf.append(feature)

    if verbose:
        if nb_inf == 0:
            print("Les données ne contiennent pas de valeur infinie")
        else:
            print(f"Les données contiennent {nb_inf} valeurs infinies"
                  f" ({float(nb_inf)/(df.shape[0] * df.shape[1]):.3f}%)"
                  f" pour les features {col_with_inf}")

    del col_with_inf
    gc.collect()

    if replace_with_nan:
        if verbose:
            print("Les valeurs infinies sont remplacées par np.NaN")
        return df.replace([np.inf, -np.inf], np.NaN)
    else:
        return nb_inf>0


def outliers(data, method='best', strategy='id_number', value=0.0):
    """
    Examine data à la recherche d'outlier selon plusieurs méthodes
        et remplace éventuellement les outliers par de nouvelles
        valeurs selon la stratégie choisie.
    :param data: array, contenant la liste des valeurs de la donnée.
    :param method: str, methode de détection des outliers
        'std': moyenne et écart type, suppose la distribution normale
        'iq': interquantile si la distribution n'est pas normale
        default='best', teste si la distribution est normale et
        applique la stratégie 'std' si oui et 'iq' sinon
    :param strategy: stratégie d'identification, voire de traitement
        des outliers.
        - Identification d'outliers: 'id_number' (nombre), 'id_rate'
        (taux), 'id_index_lower' (index outliers bas), 'id_index_upper'
        (index outliers haut), 'id_index' (index tous outliers).
        - Remplacement des outliers: 'replace_min_max' (si sous le seuil
        bas, valeur du seuil bas et vice-versa pour seuil haut),
        'replace_mean' (moyenne), 'replace_median' (médiane),
        'replace_value' (valeur=value), 'replace_nan' (np.NaN).
    :param value: float, valeur de remplacement pour la stratégie
    'replace_value'.
    :return: int (nombre d'outliers), float (taux d'outliers), ou
        list (index concernés de data) si stratégie d'identification.
        (int, array), si stratégie de remplacement avec (nombre
        d'outliers, données modifiées). Note: les données modifiées
        ne modifient pas les données d'entrée.
    """
    with np.errstate(all='ignore'):
        mean = np.nanmean(data)
        std = np.nanstd(data)
        median = np.nanmedian(data)

    # Test de distribution normale de data avec le test de Shapiro
    #if method=='best':
    #    from random import sample
    #    sample_data = sample(sorted(data), 5000) if len(data)>5000 else data
    #    stat, p = st.shapiro(sample_data)
    #    method = 'std' if p > 0.05 else 'iq'

    # Test de distribution normale de data avec le test K² d'Agostino
    if method=='best':
        stat, p = st.normaltest(data)
        method = 'std' if p > 0.05 else 'iq'

    # Calcule les bornes lower et upper selon la méthode
    if method=='std':
        lower, upper = mean - 3 * std, mean + 3 * std
    elif method=='iq':
        q25, q75 = np.nanpercentile(data, 25), np.nanpercentile(data, 75)
        lower, upper = q25 - 1.5 * (q75 - q25), q75 + 1.5 * (q75 - q25)

    index_lower = np.where(np.logical_and(data < lower, ~np.isnan(data)))[0]
    index_upper = np.where(np.logical_and(data > upper, ~np.isnan(data)))[0]
    outliers_nb = len(index_lower) + len(index_upper)

    if strategy=='id_number':
        return outliers_nb
    elif strategy=='id_rate':
        return float(outliers_nb) / len(data)
    elif strategy=='id_index_lower':
        return index_lower
    elif strategy=='id_index_upper':
        return index_upper
    elif strategy=='id_index':
        return np.sort(np.concatenate([index_lower, index_upper]))

    elif 'replace_' in strategy:
        data_repl = np.copy(data)
        index_all = np.sort(np.concatenate([index_lower, index_upper]))
        if strategy=='replace_min_max':
            data_repl[index_lower] = lower
            data_repl[index_upper] = upper
        elif strategy=='replace_mean':
            data_repl[index_all] = mean
        elif strategy=='replace_median':
            data_repl[index_all] = median
        elif strategy=='replace_value':
            data_repl[index_all] = value
        elif strategy=='replace_nan':
            data_repl[index_all] = np.NaN
        return outliers_nb, data_repl

    else:
        print("Erreur sur la valeur de 'method' ou 'strategy'")


def features_with_nan(df, num_pattern_list=None, cat_pattern_list=None, verbose=True):
    """
        Pour chaque feature de df, caractérise les NaN et recommande une stratégie de traitement.
        :param df: dataframe, dont on veut caractériser les NaN.
        :param num_pattern_list: list, chaines de caractères dans le nom de la feature signifiant
            qu'il s'agit d'une feature numérique ; default = None
        :param cat_pattern_list: list, chaines de caractères dans le nom de la feature signifiant
            qu'il s'agit d'une feature catégorielle ; default = None
        :param verbose: bool, mode verbose ; default = True
        :return: dataframe, table des NaN précisant pour chaque feature:
            - 'feature': nom de la feature
            - 'nan_nb': nombre de NaN
            - 'nan_rate': taux de NaN
            - 'type': type de variable ('num', 'cat_bin_num', 'cat_mul_num', 'cat_bin_str', 'cat_mul_tr')
            - 'nunique': nombre de valeurs uniques
            - 'unique': valeurs uniques
            - 'recommended strategy': stratégie recommandée de traitement des NaN
        """
    # Initialisations
    df_nan = None
    nuniq_max = min(0.01 * len(df), 100)
    features_list = ['feature', 'nan_nb', 'nan_rate', 'nan_minclass_rate', 'type', 'nunique', 'unique', 'recommended_strategy']
    for feature in df.columns.tolist():
        nan_nb = df[feature].isnull().sum(axis=0)
        if nan_nb>0:

            # Caractéristiques de la feature
            nan_rate = 100.0 * df[feature].isnull().sum(axis=0) / len(df)
            nan_min_class_rate = 100.0 * df.loc[df[feature].isnull() & df['TARGET'] == 1, 'TARGET'].sum() / nan_nb
            type_feat = 'num' if np.issubdtype(df[feature].dtype, np.number) else 'cat'
            nuniq = df[feature].nunique() # exclut NaN

            # Noms de feature contenant un str signifiant que la feature est numérique ou catégorielle
            if num_pattern_list is not None:
                contain_num_pattern = True if any(pattern in feature for pattern in num_pattern_list) else False
            else:
                contain_num_pattern = False
            if cat_pattern_list is not None:
                contain_cat_pattern = True if any(pattern in feature for pattern in cat_pattern_list) else False
            else:
                contain_cat_pattern = False

            # Type de variable catégorielle: 'cat' + ('_bin' ou '_mul') + ('_num' ou '_str')
            if type_feat=='cat' or nuniq<nuniq_max or contain_cat_pattern:
                uniq = df[feature].dropna().unique()
                if np.issubdtype(uniq.dtype, np.number) and contain_num_pattern:
                    type_feat = 'num'
                    uniq = np.inf
                else:
                    class_type = '_bin' if nuniq==2 else '_mul'
                    alpha_type = '_num' if np.issubdtype(uniq.dtype, np.number) else '_str'
                    type_feat = 'cat' + class_type + alpha_type
            else:
                uniq = np.inf

            # Stratégie recommandée pour le traitement des NaN
            if 'cat' in type_feat:
                strategy = 'NaN as a category'
            else:
                if nuniq > nuniq_max:
                    strategy = 'mean' if (outliers(df[feature], strategy='id_rate') < 0.01) else 'median'
                else:
                    strategy = 'most_frequent'

            # Renseignement de la table des NaN
            info_list= [feature, nan_nb, nan_rate, nan_min_class_rate, type_feat, nuniq, uniq, strategy]
            if df_nan is None:
                df_tmp = pd.DataFrame([info_list], columns=features_list)
                df_nan = df_tmp.copy()
            else:
                df_tmp = pd.DataFrame([info_list], columns=features_list)
                df_nan = pd.concat([df_nan, df_tmp], axis=0, ignore_index=True)

    # Formatage et affichage optionnel de la table des NaN
    df_nan['nan_rate'] = df_nan['nan_rate'].map('{:.1f}%'.format)
    df_nan['nan_minclass_rate'] = df_nan['nan_minclass_rate'].map('{:.2f}%'.format)
    if verbose:
        print("Caractérisation des features contenant des valeurs manquantes:")
        display(df_nan)

    # Nettoyage des variables
    del nuniq_max, features_list, nan_nb, nan_rate, type_feat, nuniq
    del contain_num_pattern, contain_cat_pattern, uniq, df_tmp
    gc.collect()
    return df_nan


def nan_treament_decisions(df_nan, nan_decisions=None, save=None):
    """
    Intègre les décisions de traitements spécifiques des valeurs
        manquantes par feature dans le dataframe df_nan.
    :param df_nan: dataframe issue de la fonction features_with_nan
    :param nan_decisions: dict, associe aux features sélectionnées
        une décision spécifique ({'feature': 'decision'}).
        Si la dénomination de la feature commence par '#' alors
        la décision est appliquée à tous les noms de features
        contenant la chaine de caractère après le '#'.
    :param save: str, nom de fichier de sauvegarde, defaut=None,
        pas de sauvegarde.
    :return: dataframe df_nan augmenté de 2 colonnes:
        - 'decision': bool, si une décision spécifique est spécifiée
        - 'nan_treatment': le traitement spécifié s'il existe
        sinon le traitement recommandé ('recommended_strategy').
    """
    # Table des décisions de traitement des NaN
    df_nan.set_index(keys='feature', drop=False, inplace=True)
    list_nan_features = df_nan['feature'].values.tolist()
    list_nan_decisions = nan_decisions.keys()
    df_nan['decision'] = False
    df_nan['nan_treatment'] = 'uncovered'
    for feat in list_nan_features:
        if feat in list_nan_decisions:
            df_nan.at[feat, 'decision'] = True
            df_nan.at[feat, 'nan_treatment'] = nan_decisions[feat]
        else:
            df_nan.at[feat, 'nan_treatment'] = df_nan.at[feat, 'recommended_strategy']
            for item in list_nan_decisions:
                if (item[0] == '#') and (item[1:] in feat):
                    df_nan.at[feat, 'decision'] = True
                    df_nan.at[feat, 'nan_treatment'] = nan_decisions[item]
                    break

    # Sauvegarde éventuelle
    if save is not None: df_nan.to_csv(save, sep=';', index=False)

    # Affichage du résultat
    uncovered = 0 if 'uncovered' not in df_nan['nan_treatment'].tolist() else df_nan['nan_treatment'].value_counts()[
        'uncovered']
    print(f"{uncovered} feature non couverte{'' if uncovered == 0 else ':'} "
          f"{', '.join(df_nan.index[df_nan['nan_treatment'] == 'uncovered'].tolist())}")
    decision_rate = df_nan.loc[df_nan['decision'], 'importance'].sum() / df_nan['importance'].sum()
    print(f"Taux de décisions spécifiques relatif à l'importance des features: {100 * decision_rate:.2f}%")
    df_nan.reset_index(drop=True, inplace=True)

    del nan_decisions, list_nan_features, feat, item, uncovered, decision_rate
    gc.collect()
    return df_nan


def nan_treatment(df, df_nan, mode='auto', save=None):
    """
    Traite les valeurs manquantes de df avec les consignes contenues
        dans df_nan.
    :param df: dataframe, avec valeurs manquantes à traiter.
    :param df_nan: dataframe, contenant les features 'feature'
        (noms des features de df), 'recommended_strategy' (pour le
         mode='auto') et 'nan_treatment' (consigne de traitement
         pour le mode='forced').
    :mode: str, choix de l'alternative de traitement:
        - 'auto': utilise la stratégie recommandée ('recommended_strategy')
        - 'forced': utilise les décisions ('nan_treatment')
    :param save: nom du fichier de sauvegarde du dataframe df traité
        ('.csv'), default=None: pas de sauvegarde.
    :return: dataframe, copie profonde de df avec valeurs manquantes
        traitées (df n'est pas modifié).
    """
    df_tmp = df.copy()
    df_nan.set_index(keys='feature', drop=False, inplace=True)
    nan_treat_feature = 'recommended_strategy' if mode=='auto' else 'nan_treatment'
    for feature in df_nan['feature']:
        nan_treat = df_nan.at[feature, nan_treat_feature]
        if nan_treat=='drop':
            print(f"Feature {feature}: suppression de {df_tmp[feature].isnull().sum()} individus")
            df_tmp.dropna(subset=feature, inplace=True)
        elif nan_treat=='max':
            df_tmp[feature].fillna(value=df_tmp[feature].max(), inplace=True)
        elif nan_treat=='mean':
            df_tmp[feature].fillna(value=df_tmp[feature].mean(), inplace=True)
        elif nan_treat=='median':
            df_tmp[feature].fillna(value=df_tmp[feature].median(), inplace=True)
        elif nan_treat=='min':
            df_tmp[feature].fillna(value=df_tmp[feature].min(), inplace=True)
        elif nan_treat=='most_frequent':
            df_tmp[feature].fillna(value=df_tmp[feature].value_counts().idxmax(), inplace=True)
        elif (nan_treat==0) or (nan_treat=='0'):
            df_tmp[feature].fillna(value=0, inplace=True)
        elif (nan_treat==1) or (nan_treat=='1'):
            df_tmp[feature].fillna(value=1, inplace=True)
        else:
            print(f"Cas de traitement non couvert: '{nan_treat}'")

    df_nan.reset_index(drop=True, inplace=True)
    if save is not None: df_tmp.to_csv(save, sep=';', index=False)
    del feature, nan_treat
    gc.collect()
    return df_tmp


def normalization_info(df, save=None, verbose=True):
    """
    Donne, sous forme d'un dataframe, les informations relative
    au besoin de normaliser 'df'.
    :param df: dataframe, contenant les données des features
        numériques du jeu de données.
        Note: pour éviter la fuite de données dans l'utilisation
        du résultat de cette fonction, 'df' devrait ne contenir
        que les données du jeu d'entrainement.
    :param save: str, chemin complet du fichier de sauvegarde
        du dataframe contenant les informations ; default=None,
        pas de sauvegarde.
    :param verbose: mode verbose.
    :return: dataframe, bool
        - dataframe contenant les informations de normalisation
        de la série de données de chaque feature de 'df':
         → 'feature': str, nom de la feature numérique
         → 'gauss': bool, si la distribution des données suit
         une loi normale
         → 'amplitude': float, max-min
         → 'min': float, min
         → 'max': float, max
         → 'variation_coef': float, std/mean
         → 'skewness': float, asymétrie
         → 'kurtosis': float: aplatissement
         → 'skew_treatment': bool, si l'asymétrie a besoin
         d'être traitée
         → 'outliers_rate': float, taux d'outliers en %
         → 'outliers_treatment': bool, s'il faut surveiller
         le besoin de traiter les outliers après normalisation
         → 'normalization': fonction de normalisation recommandée
        - bool indiquant le besoin de normaliser le jeu de
        données.
    """
    # Attention df ne doit contenir que des données d'apprentissage
    df_norm = None
    outliers_thr = min(0.01, 100.0 / len(df))
    skew_thr = 1 # <0.5 pour normalité et modérément asymétrique pour 0.5<skew<1
    kurt_thr = 3 # <3 pour normalité
    features_list = ['feature', 'gauss', 'amplitude', 'min', 'max', 'variation_coef',
                     'nunique', 'skewness', 'kurtosis', 'skew_treatment', 'outliers_rate',
                     'outliers_treatment', 'normalization']
    for feature in df.columns.tolist():
        # Distribution normale ou pas
        stat, p = st.normaltest(df[feature].values)
        normal_distrib = True if p > 0.05 else False
        normalization = 'StandardScaler' if normal_distrib else 'RobustScaler'

        # Amplitude, min, max, nunique des données d'entrée - voir synthèse sur l'ensemble des features
        data_min = df[feature].min()
        data_max = df[feature].max()
        amplitude = data_max - data_min
        variation_coef = round(df[feature].std() / df[feature].mean(), 2) if df[feature].mean() != 0 else np.NaN
        nunique = df[feature].nunique()

        # Dissymétrie avec forte amplitude
        skewness = round(df[feature].skew(), 2)
        kurtosis = round(df[feature].kurtosis(), 2)
        kurt_toohigh = True if kurtosis > kurt_thr else False
        skew_treatment = True if abs(skewness) > skew_thr and kurt_toohigh else False

        # Outliers
        outliers_rate = round(100.0 * outliers(df[feature], method='best', strategy='id_rate'), 2)
        outliers_treatment = True if not skew_treatment and kurt_toohigh and outliers_rate > outliers_thr else False

        # Enregistrement des informations dans la table
        info_list = [feature, normal_distrib, amplitude, data_min, data_max,
                     variation_coef, nunique, skewness, kurtosis, skew_treatment,
                     outliers_rate, outliers_treatment, normalization]
        if df_norm is None:
            df_tmp = pd.DataFrame([info_list], columns=features_list)
            df_norm = df_tmp.copy()
        else:
            df_tmp = pd.DataFrame([info_list], columns=features_list)
            df_norm = pd.concat([df_norm, df_tmp], axis=0, ignore_index=True)

    # Dispersion d'amplitude des données d'entrée
    #df_norm.set_index(keys='feature', drop=False, inplace=True)
    q25, q75 = np.percentile(df_norm['amplitude'], 25), np.percentile(df_norm['amplitude'], 75)
    lower, upper = q25 - 1.5 * (q75 - q25), q75 + 1.5 * (q75 - q25)
    feat_lower_list = df_norm.loc[df_norm['amplitude'] < lower, 'feature'].tolist()
    feat_upper_list = df_norm.loc[df_norm['amplitude'] > upper, 'feature'].tolist()
    scaling_required = True if len(feat_lower_list) + len(feat_upper_list) > 0 else False
    if verbose and scaling_required and len(feat_lower_list)>0:
        print(f"Traitement complémentaire de {len(feat_lower_list)} features qui ont une amplitude atypiquement")
        max_list = min(10, len(feat_lower_list))
        print("-", '\n- '.join(feat_lower_list[:max_list]), '\n')
    if verbose and scaling_required and len(feat_upper_list)>0:
        print(f"{len(feat_upper_list)} features ont une amplitude atypiquement haute:")
        max_list = min(10, len(feat_upper_list))
        print("-", '\n- '.join(feat_upper_list[:max_list]), '\n')

    # Amplitude relative à l'amplitude médiane
    df_norm['relative_amplitude'] = 100.0 * df_norm['amplitude'] / df_norm['amplitude'].median()
    df_norm['relative_amplitude'] = df_norm['amplitude'].apply(lambda x: np.round(x, 0))

    # Sauvegarde et affichage optionnels de la table des NaN
    if save is not None: df_norm.to_csv(save, sep=';', index=False)
    if verbose:
        print("Caractérisation des features en vue de leur normalisation:")
        display(df_norm.head())

    # Nettoyage des variables
    del df_tmp
    gc.collect()
    return df_norm, scaling_required


from scipy.stats import skew
def skew_treatment(data, val_range=None, train_indexes=None, feat_name=None, max_iter=1000, verbose=False):
    """
    Transforme les données 'data' avec un skew élevé pour se rapprocher
        d'une loi normale en minimisant skew. Transformation inverse:
        - skew > 0: data = data_min + eps + eps * exp(data_transformed)
        - skew < 0: data = data_max + eps - eps * exp(data_transformed)
    :param data: numpy array, données d'entrée à transformer.
    :param val_range: tuple, (min(data), max(data)), default=None calcule
        min et max sur l'ensemble complet des données 'data' (légère
        fuite de données) → nécessaire pour le calcul du log.
    :param train_indexes: numpy array, liste des index du jeu d'entrainement
        default=None, toutes les valeurs de data sont prises en compte
    :param feat_name: str, nom optionnel de feature pour 'data'
    :param max_iter: int, nombre maximum d'itération de l'algorithme de
        transformation.
    :return: bool, numpy array, dict
        - err: indique si erreur de convergence
        - data_transformed: données transformées
        - skew_param: dict, paramètres de transformation:
            → sk_right_skewed: bool, indique le sens de la queue de 'data'
            → sk_eps: float, paramètre optimisé (skew=0) de la fonction
            de transformation
            → sk_data_min: float, paramètre (min(data)) de la fonction
            de transformation
            → sk_data_max: float, paramètre (max(data)) de la fonction
            de transformation
    """
    # Initialisations
    if verbose: print(f"Traitement de l'asymétrie de {feat_name}")
    train_indexes = np.arange(len(data)) if train_indexes is None else train_indexes
    data_train = data[train_indexes]
    err = True
    right_skewed = True if skew(data_train) >= 0 else False
    data_min = val_range[0] if val_range is not None else np.array(data).min()
    data_max = val_range[1] if val_range is not None else np.array(data).max()

    eps0 = (data_max - data_min) / 1000
    if right_skewed:
        sk0 = skew(np.log((np.array(data_train) - data_min + eps0) / eps0))
    else:
        sk0 = skew(np.log((data_max - np.array(data_train) + eps0) / eps0))
    sk = [sk0]
    ep = [eps0]
    eps1 = 2 * eps0

    # Algorithme de convergence: eps pour minimiser abs(skew)
    for n_iter in range(1, max_iter+1):
        if right_skewed:
            sk1 = skew(np.log((np.array(data_train) - data_min + eps1) / eps1))
        else:
            sk1 = skew(np.log((data_max - np.array(data_train) + eps1) / eps1))
        ep.append(eps1)
        sk.append(sk1)
        if abs(sk1) > 0.1:
            if sk0 * sk1 < 0:
                eps = (eps0 + eps1) / 2
            else:
                alpha = abs(sk1 - sk0) * abs(sk1)
                eps = eps0 if sk1 * (sk1 - sk0) > 0 else eps1
                if (sk1 - sk0) * (eps1 - eps0) > 0:
                    alpha = min(0.9, alpha) if sk1>0 else min(10, alpha)
                    eps = eps * (1 - sk1 / abs(sk1) * alpha)
                else:
                    alpha = min(0.9, alpha) if sk1 < 0 else min(10, alpha)
                    eps = eps * (1 + sk1 / abs(sk1) * alpha)
            eps = max(eps, 1e-20)
            eps0 = eps1
            eps1 = eps
            if ((eps0==1e-20) and (eps1==1e-20)) or (sk1==sk0):
                break
        else:
            err = False
            if verbose:
                print(f"Convergence à la {n_iter}{'ème' if n_iter > 1 else 'ère'} itération")
            # Paramètres de transformation des données
            skew_param = {'sk_right_skewed': right_skewed,
                          'sk_eps': eps1,
                          'sk_data_min': data_min,
                          'sk_data_max': data_max}
            break

    # Si le skew n'a pas pu être réduit au seuil spécifié
    if err == True:
        eps1 = ep[np.argmin(np.array(sk))]
        skew_param = {'sk_right_skewed': right_skewed,
                      'sk_eps': eps1,
                      'sk_data_min': data_min,
                      'sk_data_max': data_max}

    # Courbes skew et eps en fonction des itérations
    if verbose:
        if err==True: print(f"-> Le skew n'a pas pu être réduit au dessous du seuil")
        sk = np.array(sk)
        ep = np.array(ep)
        fig, (ax1, ax3) = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
        ax1.plot(np.arange(len(sk)), sk, color='steelblue', label='skew')
        ax1.set_xlabel('itération', fontsize=12)
        ax1.set_ylabel('skew', color='steelblue', fontsize=14)
        ax2 = ax1.twinx()
        ax2.plot(np.arange(len(ep)), ep, color='coral', label='eps')
        ax2.set_ylabel('eps', color='coral', fontsize=14)

        ax3.hist(data, bins=100, color='steelblue')
        ax3.hist(data_train, bins=100, color='coral')
        ax3.set_xlabel('histogramme', fontsize=12)

        if feat_name is not None: plt.suptitle(feat_name, fontsize=14)
        plt.subplots_adjust(wspace=0.3)
        plt.show()


    # Transformation des données d'entrainement + test
    if right_skewed:
        data_transformed = np.log((np.array(data) - data_min + eps1) / eps1)
    else:
        data_transformed = np.log((data_max - np.array(data) + eps1) / eps1)
    if verbose:
        print(f"Evolution du skew de {skew(data_train)} vers {skew(data_transformed[train_indexes])}")

    # Nettoyage des données
    del data_train, sk0, sk1, n_iter, sk, ep, eps0
    gc.collect()

    return err, data_transformed, skew_param


def data_gen(nbr=100, seed=0, gauss=True, mean=0, std=1, sk=0.5, minmax=(0,1), right_skew=True):
    """
    Génère une matrice de ('nbr') nombres aléatoires ('seed'):
        - soit gaussienne (paramètres 'mean' et 'std'),
        - soit asymétrique (paramètres 'sk', 'minmax' et
        'right_skew').
    :param nbr: int, taille de la matrice de sortie.
    :param seed: int, seed du générateur aléatoire.
    :param gauss: bool, si les points sont générés selon une
        distribution gaussienne ou pas.
    :param mean: float, spécifier la moyenne s'il est attendu
        une distribution gaussienne.
    :param std: float,spécifier l'écart type s'il est attendu
        une distribution gaussienne.
    :param sk: float pour une distribution non gaussienne,
        paramètre influent pour le skewness et kurtosis, qui
        augmentent avec sk.
    :param minmax: (float, float), pour une distribution non
        gaussienne, spécifie les bornes des données de sortie
    :param right_skew: bool, pour une distribution non gaussienne,
        spécifie si la queue est à droite (True) ou gauche
        (False).
    :return: numpy.array, matrice de taille 'nbr' des données.
    """
    np.random.seed(seed)
    X = np.random.randn(nbr)
    if gauss:
        X = mean + std * np.random.randn(nbr)
    else:
        X = (np.exp(sk * X) - 1)
        if right_skew:
            a = float((minmax[1] - minmax[0])) / (X.max() - X.min())
            b = minmax[1] - a * X.max()
        else:
            a = -float((minmax[1] - minmax[0])) / (X.max() - X.min())
            b = minmax[1] - a * X.min()
        X =  a * X + b
        del a, b
        gc.collect()
    return X


from sklearn.preprocessing import StandardScaler, RobustScaler
def df_normalization(df, df_norm, train_indexes=None, save_df=None, save_dfnorm=None, verbose=1):
    """
    Normalise le jeu de données contenu dans 'df' (features
        de 'df_norm' seulement), avec les informations de 'df_norm'.
        Afin d'éviter les fuites de données (implication du jeu
        de test), les paramètres des fonctions de transformation
        sont calculés sur le jeu d'entrainement, correspondant
        aux indices  'train_indexes' de 'df'. Les paramètres de
        transformation sont spécifiés dans 'df_norm' qui peut
        être sauvegardé.
        Transformations inverses (chaque transformation est
        applicable si ses paramètres existes), dans l'ordre:
        - Amplitude : X' = X / ampl_coef
        - RobustScaler: X' = (rsp_q3-rsp_q1) * X + rsp_q2
        - StandardScaler: X' = ssp_std * X - ssp_mean
        - Skew, selon sk_right_skewed:
            True: X' = sk_data_min + sk_eps + sk_eps * exp(X)
            False: X' = sk_data_max + sk_eps - sk_eps * exp(X)
    :param data: numpy array, données d'entrée à transformer.
    :param df: dataframe, jeu de données à normaliser.
    :param df_norm: dataframe, informations de normalisation
        de df retourné par 'normalization_info'.
    :param train_indexes: array, liste des index du jeu
        d'entrainement. Note: l'index de df doit être numéroté
        de 0 à len(df)-1.
    :param save_df: str, chemin complet du fichier de
        sauvegarde de df transofrmé ; default=None, pas de
        sauvegarde.
    :param save_dfnorm: str, chemin complet du fichier de
        sauvegarde de df_norm (tq modifié par cette fonction
        pour inclure les paramètres de transformation) ;
        default=None, pas de sauvegarde.
    :param verbose: int, niveaux:
        - 0: off
        - 1: verbose sauf pour 'skew_treatment'
        - 2: full verbose, y compris 'skew_treatment'
    :return: dataframe, 'df' transformé normalisé sur les
        features de 'df_norm'.
    """
    # Initialisations
    full_verbose = True if verbose==2 else False
    verbose = True if verbose>0 else False
    df_normalized = df.copy()
    df_norm.set_index(keys='feature', drop=False, inplace=True)
    train_indexes = np.arange(len(df)) if train_indexes is None else train_indexes
    num_features = df_norm['feature'].tolist()

    # Traitement des asymétries et mises à l'échelle
    with tqdm(total=len(num_features), position=0, leave=True, desc='Normalisation') as pbar:
        count_skt = 0
        count_skp = 0
        for feat in num_features:
            if df_norm.at[feat, 'skew_treatment']:
                count_skt += 1
                err, data_transformed, skew_param = \
                    skew_treatment(df[feat], train_indexes=train_indexes, feat_name=feat, verbose=full_verbose)
                if not err:
                    df_normalized[feat] = data_transformed
                    for param in skew_param.keys():
                        if param not in df_norm.columns: df_norm[param] = None
                        df_norm.at[feat, param] = skew_param[param]
                else:
                    count_skp += 1
            if df_norm.at[feat, 'normalization'] == 'StandardScaler':
                scaler = StandardScaler()
                ss_param = {'ssp_mean': df_normalized.loc[train_indexes, feat].mean(),
                            'ssp_std': df_normalized.loc[train_indexes, feat].std()}
                for param in ss_param.keys():
                    if param not in df_norm.columns: df_norm[param] = None
                    df_norm.at[feat, param] = ss_param[param]
            else:
                scaler = RobustScaler()
                rs_param = {'rsp_q1': np.quantile(df_normalized.loc[train_indexes, feat].values, 0.25),
                            'rsp_q2': np.quantile(df_normalized.loc[train_indexes, feat].values, 0.5),
                            'rsp_q3': np.quantile(df_normalized.loc[train_indexes, feat].values, 0.75)}
                for param in rs_param.keys():
                    if param not in df_norm.columns: df_norm[param] = None
                    df_norm.at[feat, param] = rs_param[param]
            scaler.fit(df_normalized.loc[train_indexes, feat].values.reshape(-1, 1))
            df_normalized[feat] = scaler.transform(df_normalized[feat].values.reshape(-1, 1))
            pbar.update()
    if verbose:
        print(f"Normalisation effectuée, dont {count_skp} réductions "
              f"partielles de l'asymétrie sur {count_skt}\n")

    # Vérification de la mise à l'échelle
    df_norm_verif, scaling_required = normalization_info(df_normalized.loc[train_indexes, num_features],
                                                         verbose=full_verbose)

    # Traitement complémentaire éventuel de mise à l'échelle
    if scaling_required:
        df_norm_verif.set_index(keys='feature', drop=False, inplace=True)
        q25, q75 = np.percentile(df_norm_verif['amplitude'], 25), np.percentile(df_norm_verif['amplitude'], 75)
        lower, upper = q25 - 1.5 * (q75 - q25), q75 + 1.5 * (q75 - q25)
        feat_lower_list = df_norm_verif.loc[df_norm_verif['amplitude'] < lower, 'feature'].tolist()
        feat_upper_list = df_norm_verif.loc[df_norm_verif['amplitude'] > upper, 'feature'].tolist()
        if verbose and len(feat_lower_list)>0:
            print(f"Traitement résiduel sur l'amplitude atypiquement basse de {len(feat_lower_list)} features")
            if full_verbose:
                print("-", '\n- '.join(feat_lower_list), '\n')
        for feat in feat_lower_list:
            df_normalized[feat] = df_normalized[feat] * lower / df_norm_verif.at[feat, 'amplitude']
            df_norm.at[feat, 'ampl_coef'] = lower / df_norm_verif.at[feat, 'amplitude']
        if verbose and len(feat_upper_list)>0:
            print(f"Traitement résiduel sur l'amplitude atypiquement haute de {len(feat_upper_list)} features")
            if full_verbose:
                print("-", '\n- '.join(feat_upper_list), '\n')
        for feat in feat_upper_list:
            df_normalized[feat] = df_normalized[feat] * upper / df_norm_verif.at[feat, 'amplitude']
            df_norm.at[feat, 'ampl_coef'] = upper / df_norm_verif.at[feat, 'amplitude']
        df_norm_verif, scaling_required = normalization_info(df_normalized.loc[train_indexes, num_features],
                                                             verbose=full_verbose)
        del q25, q75, feat_lower_list, feat_upper_list

    # Résultat final du processus de normalisation
    print("Normalisation terminée avec succès:", scaling_required)
    df_norm_verif.set_index(keys='feature', drop=False, inplace=True)
    df_norm['relative_amplitude'] = df_norm_verif['relative_amplitude'].copy()

    # Sauvegarde éventuelle du jeu de données normalisé et table de normalisation
    if save_df is not None:
        df_normalized.to_csv(save_df, sep=';', index=False)
    if save_dfnorm is not None:
        df_norm.to_csv(save_dfnorm, sep=';', index=False)

    # Nettoyage des variables
    if len(num_features)>0: del count_skt, count_skp, feat, scaler, param
    if 'err' in locals(): del err, data_transformed, skew_param
    if 'ss_param' in locals(): del ss_param
    if 'rs_param' in locals(): del rs_param
    del df_norm, num_features, df_norm_verif, scaling_required
    gc.collect()

    return df_normalized


import re
import random
def load_dataset(set='forced', subset='train', subset_size=1.0, debug=False):
    """
    Charge le dataset dans un dataframe.
    :param set: str, nom de la version du dataset:
        - 'auto': dataset avec nan traitées automatiquement
        - 'forced': dataset avec nan traitées avec décisions
    :param subset: str, default='train':
        - 'train': jeu avec 'TARGET'=NaN
        - 'test': jeu avec 'TARGET'!=NaN
    :param subset_size: float ]0.0, 1.0], default=1.0,
        utilisé seulement si subset='test',
        proportion du jeu de test à charger.
    :param debug: bool, mode debug
    :return: dataframe, Series:
        - X: dataframe, matrice des entrées X,
        - y: Series, matrice des classes vraies.
    """
    fname = data_path + 'P7_data_preprocessed_woNaN_forced_normalized.csv' if set=='forced'\
        else data_path + 'P7_data_preprocessed_woNaN_auto_normalized.csv'
    nrows = 10000 if debug else None
    df = pd.read_csv(fname, sep=';', nrows=nrows)
    # Pour éviter "LightGBMError: Do not support special JSON characters in feature name"
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    # Features d'entrée de la prédiction
    non_input_features_list = ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']
    x_columns = [f for f in df.columns.tolist() if f not in non_input_features_list]
    # Type categorical pour la prédiction avec lightGBM
    with open(data_path + 'P7_cat_features.txt', "r") as file:
        categorical_columns = json.load(file)
    categorical_columns = [re.sub('[^A-Za-z0-9_]+', '', f) for f in categorical_columns if
                           f not in non_input_features_list]
    for col in categorical_columns:
        df[col] = pd.Categorical(df[col])
    # production des sorties selon 'subset'
    if subset=='train':
        train_indexes = df.index[~df['TARGET'].isnull()]
        X = df.loc[train_indexes, x_columns].copy()
        y = df.loc[train_indexes, 'TARGET'].copy()
        cat_index = [X.columns.get_loc(col) for col in categorical_columns]
        print(f"Dimensions du jeu de données: X = {X.shape}, y = {y.shape}\n")
        del df, non_input_features_list, x_columns, train_indexes
        gc.collect()
        return X, y, cat_index
    else:
        indexes = df.index[df['TARGET'].isnull()].tolist()
        idx = random.sample(indexes, int(subset_size*len(indexes)))
        df = df.loc[idx, :].sort_values(by='SK_ID_CURR', ignore_index=True)
        X = df.loc[:, x_columns].copy()
        print(f"Dimensions du jeu de données: X = {X.shape}\n")
        id_list = list(df['SK_ID_CURR'])
        del df, non_input_features_list, x_columns, indexes, idx
        gc.collect()
        return X, id_list


import sklearn
import lightgbm
print("Versions des librairies des modèles:")
print('- Scikit-learn : ' + sklearn.__version__)
print('- LightGBM : ' + lightgbm.__version__)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import get_scorer_names
import time
def model_evaluation(X, y, model, metric='average_precision',
                     n_splits=5, n_repeats=3,
                     random_state=None, n_jobs=-1, verbose=False):
    """
    Evalue le modèle en terme de scores et temps de fit+évaluation
        unitaire.
    :param X: Dataframe, matrice d'entrée.
    :param y: Series, matrice des classes vraies.
    :param model: instance de modèle.
    :param metric: str ou scorer, métrique sous forme de str ou telle
        que renvoyée par la fonction sklearn.metrics.make_scorer.
    :param n_splits: int, nombre de découpe du dataset.
    :param n_repeats: int, nombre de répétition de découpe du dataset.
    :param random_state: int ou None.
    :param n_jobs: int ou None, nombre de processus exécutés en
        parallèle ; default=-1, maximum.
    :param verbose: bool, mode verbose.
    :return: numpy array, float
        - scores: array, matrice des n_repeats * n_splits scores
        - exec_time: float, temps d'exécution unitaire.
    """
    if (type(metric) == str) and (metric not in get_scorer_names()):
        print("Le nom de la métrique n'est pas valide")
        return
    start_time = time.time()
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    v = 11 if verbose else 0
    scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=n_jobs, verbose=v)
    elapsed = time.time() - start_time
    if verbose:
        print(f"→ Evaluation exécutée en {elapsed_format(elapsed)}")
    exec_time = elapsed/(n_splits*n_repeats)
    del cv, start_time, elapsed
    gc.collect()
    return scores, exec_time

def display_models_eval(model_list):
    """
    Affiche la comparaison des modèles sous forme de 2 graphiques:
        - gauche: boite à moustaches concernant les scores ;
        - droite: barres concernant les temps de calcul unitaires.
    :param model_list: list, liste des modèles à comparer, telle
        que créée par la fonction 'create_models' du Notebook.
        Chaque élément de la liste est un dictionnaire avec au
        minimum les clés:
        - 'name': str, acronyme du modèle ;
        - 'model': instance du modèle ;
        - 'scores': scores de l'évaluation tels que renvoyés par
        model_evaluation ;
        - 'time': temps d'exécution tels que renvoyés par
        model_evaluation.
    :return: pas de retour
    """
    labels = [model['name'] for model in model_list]
    scores = np.column_stack([model['eval']['scores'] for model in model_list])
    times = np.array([model['eval']['time'] for model in model_list])
    plt.figure(figsize=(15, 6))
    plt.subplot(121)
    plt.boxplot(scores, labels=labels, showmeans=True)
    plt.title("Scores")
    plt.subplot(122)
    plt.bar(x= labels, height=times)
    plt.title("Times")
    plt.suptitle("Comparaison des modèles", fontsize=14)
    plt.show()
    del labels, scores, times
    gc.collect()


import optuna
print('Version de la librairie Optuna: ' + optuna.__version__, '\n')

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
def objective(trial, model_name, X, y, resampling=True,
              metric='average_precision', n_splits=5, n_repeats=3,
              random_state=None, njobs=-1, verbose=False):
    """
    Fonction 'objective' telle que spécifiée par la librairie
        'Optuna'.
        - Définit les plages des paramètres à optimiser.
        - Applique successivement:
            . fonction de resempling (over puis under sampling)
            . modèle.
        - Calcule le score avec la fonction 'model_evaluation'.
    :param trial: paramètre d'Optuna, correspond à un essai.
    :param model_name: str, nom du modèle.
    :param X: Dataframe, matrice d'entrée.
    :param y: Series, matrice des classes vraies.
    :param resampling: bool, spécifie si des méthodes de
        resampling doivent être explorées.
    :param metric: str ou scorer, métrique sous forme de str ou telle
        que renvoyée par la fonction sklearn.metrics.make_scorer.
    :param n_splits: n_splits: int, nombre de découpe du dataset.
    :param n_repeats: int, nombre de répétition de découpe du dataset.
    :param random_state: int ou None.
    :param n_jobs: int ou None, nombre de processus exécutés en
        parallèle ; default=-1, maximum.
    :param verbose: bool, mode verbose.
    :return: float, score de l'essai ('trial').
    """
    # Plages des paramètres des modèles
    if model_name=='LR':
        lr_c = trial.suggest_float('C', 1e-10, 1e10, log=True)
        lr_cw = trial.suggest_categorical('class_weight', [None, 'balanced'])
        lr_s = trial.suggest_categorical('solver', ['saga'])
        clf = LogisticRegression(C=lr_c, class_weight=lr_cw, solver=lr_s, random_state=random_state, n_jobs=njobs)
    elif model_name=='RF':
        rf_ne = trial.suggest_int('n_estimators', 100, 200)
        rf_cw = trial.suggest_categorical('class_weight', [None, 'balanced'])
        rf_cr = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
        clf = RandomForestClassifier(n_estimators=rf_ne, class_weight=rf_cw, criterion=rf_cr, random_state=random_state, n_jobs=njobs)
    elif model_name=='LG':
        lg_ne = trial.suggest_int('n_estimators', 100, 200)
        lg_lr = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
        lg_md = trial.suggest_int('max_depth', 7, 11)
        clf = LGBMClassifier(n_estimators=lg_ne, learning_rate=lg_lr, max_depth=lg_md, random_state=random_state, n_jobs=njobs)
    else:
        print(f"model_name={model_name} n'est pas répertorié")
        return 0

    # Resampling
    if resampling:
        sampling = trial.suggest_categorical('sampling', ['None', 'ST', 'SE'])
        if sampling=='ST':
            spl = SMOTETomek(tomek=TomekLinks(n_jobs=njobs), random_state=random_state, n_jobs=njobs)
            pipe = Pipeline(steps=[('sample', spl), ('model', clf)])
        elif sampling=='SE':
            spl = SMOTEENN(enn=EditedNearestNeighbours(n_jobs=njobs), random_state=random_state, n_jobs=njobs)
            pipe = Pipeline(steps=[('sample', spl), ('model', clf)])
        else:
            pipe = Pipeline(steps=[('model', clf)])
    else:
        pipe = Pipeline(steps=[('model', clf)])

    # Calcul du score
    scores, elapsed = model_evaluation(X, y, pipe, metric=metric,
                                       n_splits=n_splits, n_repeats=n_repeats,
                                       random_state=None, n_jobs=njobs, verbose=verbose)
    if verbose:
        print(f"Score: {np.mean(scores)} ({np.var(scores)})")
    return np.mean(scores)


import json
from sklearn.decomposition import PCA
def perform_study(model_name, dataset='forced', resampling=True, pca_var=0.95,
                  metric='average_precision', n_splits=5, n_repeats=3, n_trials=1,
                  random_state=None, n_jobs=-1, debug=True, verbose=True):
    """
    Exécute une étude ('study') d'optimisation des paramètres du
        modèle, telle que spécifiée par la librairie 'Optuna'.
    :param model_name: str, nom (acronyme) du modèle concerné.
    :param dataset: str, nom du dataset concerné, qui est chargé
        par la fonction dans les matrices X et y.
    :param resampling: bool, spécifie si des méthodes de
        resampling doivent être explorées.
    :param pca_var: float compris dans l'intervale ]0, 1[,
        variance expliquée de la réduction de dimentionnalité
        avec PCA.
    :param metric: str ou scorer, métrique sous forme de str ou telle
        que renvoyée par la fonction sklearn.metrics.make_scorer.
    :param n_splits: n_splits: int, nombre de découpe du dataset.
    :param n_repeats: int, nombre de répétition de découpe du dataset.
    :param n_trials: int, nombre d'essais de l'étude.
    :param random_state:int ou None.
    :param n_jobs: int ou None, nombre de processus exécutés en
        parallèle ; default=-1, maximum.
    :param debug: bool, mode 'debug'
    :param verbose: bool, mode verbose.
    :return: objet 'study' tel que défini par la librairie 'Optuna'
    """
    mode_debug = ' - Mode debug' if debug else ''
    print(Fore.BLACK + Style.BRIGHT + Back.WHITE +
      f"Dataset '{dataset}' - Optimisation des hyperparamètres du modèle '{model_name}'{mode_debug}:\n"
      + Style.RESET_ALL)

    # Chargement du dataset et réduction de dimensionnalité
    X, y, _ = load_dataset(set=dataset, debug=debug)

    # Réduction de la dimensionalité
    # Suppression des éventuelles colonnes '_nan' (se déduisent des autres catégories)
    non_input_features_list = ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']
    with open(data_path + 'P7_cat_features.txt', "r") as file:
        categorical_columns = json.load(file)
    categorical_columns = [re.sub('[^A-Za-z0-9_]+', '', f) for f in categorical_columns if
                           f not in non_input_features_list]
    n_categorical_columns = len(categorical_columns)
    categorical_columns = [col for col in categorical_columns if col[-4:] != '_nan']
    if len(categorical_columns) < n_categorical_columns:
        print(f"Suppression de {n_categorical_columns - len(categorical_columns)}"
              f"features catégorielles correspondant aux NaN")
    # Normalisation des features catégorielles en vue du PCA
    for col in categorical_columns:
        X[col] = X[col] / np.sqrt(X[col].sum() / len(X[col])) if X[col].sum() > 0 else 0
        X[col] = X[col] - X[col].mean()
    if get_df_nan_rate(X, verbose=False) > 0:
        print("La réduction PCA ne peut s'effectuer avec des NaN: pas de réduction de dimension effectuée")
    else:
        dimX = X.shape
        pca = PCA(n_components=pca_var, random_state=random_state)
        X = pd.DataFrame(pca.fit_transform(X))
        print(f"Réduction de dimensionnalité de X (variance expliquée="
              f"{pca_var:.3f}): {dimX} → {X.shape}", '\n')

    # Définition de l'étude
    study_name = model_name.lower() + '_' + dataset + '_debug' if debug else model_name.lower() + '_' + dataset
    storage = 'sqlite:///' + data_path + study_name + '.db'
    study = optuna.create_study(study_name=study_name,
                                storage=storage,
                                load_if_exists=True,
                                direction="maximize",
                                sampler=optuna.samplers.TPESampler())
    if verbose: print(f"Sampler is {study.sampler.__class__.__name__}")

    # Exécution de l'étude
    objective_func = lambda trial: objective(trial, model_name, X, y,
                                             resampling=resampling,
                                             metric=metric,
                                             n_splits=n_splits,
                                             n_repeats=n_repeats,
                                             random_state=random_state,
                                             njobs=n_jobs,
                                             verbose=verbose)
    start_time = time.time()
    study.optimize(objective_func, n_trials=n_trials)
    elapsed = time.time() - start_time
    print(f"\n→ Recherche exécutée en {elapsed_format(elapsed)}\n")

    # Nettoyage des variables
    del X, y, mode_debug, study_name, storage, start_time, elapsed
    gc.collect()
    return study


def get_study_results(model_name, dataset='forced', debug=True, save=True, display=True):
    """
    Charge et affiche les résultats de l'étude ('study') d'optimisation
        des hyperparamètres et retourne le meilleur résultat.
    :param model_name: str, nom (acronyme) du modèle concerné.
    :param dataset: str, nom du dataset concerné, qui est chargé
        par la fonction dans les matrices X et y.
    :param debug: bool, mode 'debug'.
    :param save: bool, sauvegarde l'étude sous forme d'un fichier csv.
    :param display: bool, affichage de l'étude.
    :return: trial (tq défini par la librairie 'optuna') du meilleur
        score.
    """
    mode_debug = ' - Mode debug' if debug else ''
    print(Fore.BLACK + Style.BRIGHT + Back.WHITE +
      f"Résultat de la recherche d'hyperparamètres du modèle '{model_name}'{mode_debug}:\n"
      + Style.RESET_ALL)
    study_name = model_name.lower() + '_' + dataset + '_debug' if debug else model_name.lower() + '_' + dataset
    storage = 'sqlite:///' + data_path + study_name + '.db'

    study = optuna.study.load_study(study_name=study_name, storage=storage)
    print(f"L'étude comprend {len(study.trials)} essais")
    best_trial = study.best_trial
    print(f"score = {best_trial.value}")
    print(f"Meilleurs hyperparamètres: {best_trial.params}")

    if save:
        df_results = study.trials_dataframe(attrs=("number", "value", "params", "state"))
        df_results.to_csv(data_path + study_name + '.csv', sep=';', index=False)
        del df_results

    if display:
        #fig = optuna.visualization.plot_contour(study)
        #fig.show()
        fig = optuna.visualization.plot_optimization_history(study)
        fig.show()
        fig = optuna.visualization.plot_param_importances(study)
        fig.show()
        del fig

    del mode_debug, study_name, storage, study
    gc.collect()
    return best_trial


def del_study(model_name, dataset='forced', debug=True):
    """
    Efface l'étude correspondant au nom de modèle et mode debug.
    :param model_name: str, nom (acronyme) du modèle concerné.
    :param dataset: str, nom du dataset concerné, qui est chargé
        par la fonction dans les matrices X et y.
    :param debug: bool, mode 'debug'.
    :return: pas de retour.
    """
    study_name = model_name.lower() + '_' + dataset + '_debug' if debug else model_name.lower() + '_' + dataset
    storage = 'sqlite:///' + data_path + study_name + '.db'
    print(f"Suppression de l'étude {study_name}")
    optuna.study.delete_study(study_name=study_name, storage=storage)


from math import isnan
def get_optuna_bestparam(model, study_name):
    """
    Extrait les meilleurs hyperparamètres déterminés par Optuna
        et les ajoute sous forme de dictionnaires au modèle.
    :param model: dict, modèle tel que créé par par la fonction
        'create_models' du Notebook.
    :param study_name: str, nom de l'étude (study) réalisée avec
        Optuna.
    :return: aucun retour, 'model' est modifié pour lui ajouter:
        - dict, 'optuna_hyperparam': meilleurs hyperparamètres
        de l'étude concernant le modèle ;
        - dict, 'optuna_sampling': meilleur hyperparamètre de
        l'étude concernant l'échantillonnage.
    """
    optuna_results = pd.read_csv(data_path + study_name + '.csv', sep=';')
    optuna_results = optuna_results[optuna_results['state']=='COMPLETE'].fillna(
        '').sort_values(by='value', ascending=False, ignore_index=True)
    hyperparam_names = [name.split('params_')[1] for name in optuna_results.columns.tolist()
                        if 'params_' in name and '_sampling' not in name]
    hyperparam = dict()
    for name in hyperparam_names:
        value = optuna_results.at[0, 'params_'+name]
        if value!='':
            hyperparam[name] = value
    model['optuna_hyperparam'] = hyperparam
    model['optuna_sampling'] = optuna_results.at[0, 'params_sampling']


import shap
def shap_feature_impact(shap_values, class_value=1, top_n=10, show=True, save=None):
    """
    Calcule l'impact des features vers la classe 'class_value' et
        affiche le graphe des 'top_n' features les plus impactantes.
    :param shap_values: shap_values retournées par l'objet
        'shap.Explainer'.
    :param class_value: int, valeur de la classe cible.
    :param top_n: int, nombre de features à représenter sur le
        graphique.
    :param save: str, chemin complet du fichier pour
        l'enregistrement du graphique ; default=None, pas
        d'enregistrement.
    :return: dataframe, impact des features:
        - 'feature': nom de la feature ;
        - 'SHAP_abs': valeur absolue de la valeur de Shapley ;
        - 'Sign': signe de la valeur de Shapley ('coral': >0).
    """
    # SHAP values pour la classe et données d'entrée X sous forme de dataframe
    feature_list = shap_values.feature_names
    df_shap_val = pd.DataFrame(shap_values[..., class_value].values, columns=feature_list)
    df_X = pd.DataFrame(shap_values.data, columns=feature_list)

    # Signe de correlation entre les données et valeurs de Shapley
    corr_list = list()
    for feature in feature_list:
        corr = '#ff0051' if df_shap_val[feature].corr(df_X[feature])>0 else '#008bfb'
        corr_list.append(corr)
    corr_df = pd.concat([pd.Series(feature_list), pd.Series(corr_list)], axis=1).fillna('lightgray')
    corr_df.columns  = ['Feature','Sign']

    # Bar-graph des valeurs d'impact signées
    df_impact = pd.DataFrame(np.abs(df_shap_val).mean()).reset_index()
    df_impact.columns = ['Feature','SHAP_abs']
    df_impact = df_impact.merge(corr_df, left_on='Feature', right_on='Feature', how='inner')
    df_impact = df_impact.sort_values(by='SHAP_abs', ascending = False)
    fig, ax = plt.subplots()
    ax = df_impact[:top_n].plot.barh(x='Feature', y='SHAP_abs', ax=ax, color=df_impact['Sign'],
                                     figsize=(10, 1+min(12, int(0.2*top_n))), legend=False)
    ax.set_ylabel("")
    ax.invert_yaxis()
    ax.set_xlabel("SHAP Value", fontsize=12)
    plt.title(f"Impact des top {top_n} features sur la prédiction de la classe {class_value}\n"
              f"(Corrélation: rouge=positive, bleu=négative)", fontsize=14)
    plt.tight_layout()
    if save is not None:
        plt.savefig(save, dpi=300)
    if show:
        plt.show()
    else:
        return fig


def bivariate_cat_cat(df_cat1_cat_2, alpha=0.05, save=None):
    """
    Effectue l'analyse bivariée entre 2 variables catégorielles.
        Affiche la heatmap et effectue le test du chi2 avec un
        seuil de 5% pour évaluer la dépendance des features.
    :param df_cat1_cat_2: dataframe, contenant en ligne toutes
        les observations et 2 colonnes, une pour chaque feature.
    :param alpha: float, seuil de test de la pvalue.
    :param save: str, chemin vers le fichier d'enregistrement du
        graphique ; default=None, pas d'enregistrement.
    :return: rien
    """
    # Format des étiquettes de valeur unique
    df = df_cat1_cat_2.copy()
    features = df.columns.tolist()
    if len(features)!=2:
        print(f"Le dataframe ne correspond pas à une paire de features.")
        return
    for feature in features:
        is_feat_num = True if np.issubdtype(df[feature].dtype, np.number) else False
        if is_feat_num:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
            is_int = np.array([x%1==0 for x in pd.unique(df[feature])]).all()
            if is_int:
                df[feature] = df[feature].astype(int)

    # Table de contingence
    cont = df.pivot_table(index=features[0],
                          columns=features[1],
                          aggfunc=len,
                          margins=True,
                          margins_name='total')
    # Table ξ (xi) des corrélations
    tx = cont.loc[:,["total"]]
    ty = cont.loc[["total"],:]
    n = len(df)
    indep = tx.dot(ty) / n
    cont = cont.fillna(0)
    measure = (cont-indep)**2/indep
    xi_n = measure.sum().sum()

    # Test CHI2 (note: xi_n=chi2) - H0: variables indépendantes
    chi2, p_value, ddl, exp = st.chi2_contingency(cont)
    indep = False if p_value < alpha else True

    # Heatmap (échelle 0-1)
    table = measure/xi_n
    sns.heatmap(table.iloc[:-1,:-1],
                # valeurs de la table des contingences
                annot=cont.iloc[:-1,:-1].astype(int),
                # format de 'annot'
                fmt='d',
                cbar_kws={'label': '← independance    -    dependance →'})
    dep = 'variables non corrélées' if indep else 'variables corrélées'
    plt.title(f"Heatmap analyse bivariée ({dep})", fontsize=14)
    plt.tight_layout()
    if save is not None:
        plt.savefig(save, dpi=300)
    plt.show()

    # Nettoyage des variables
    del cont, tx, ty, n, indep, measure, xi_n, chi2, p_value, ddl, exp, table, dep
    gc.collect()


def eta_squared(x, y):
    """
    Calcul du rapport de corrélation entre une variable
        catégorielle x et une variable quantitative y.
    :param x: pandas Series, variable catégorielle.
    :param y: pandas Series, variable numérique.
    :return: float, coefficient de corrélation η²
    """
    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x == classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj - moyenne_y) ** 2 for yj in y])
    SCE = sum([c['ni'] * (c['moyenne_classe'] - moyenne_y) ** 2 for c in classes])
    eta_squared = SCE / SCT
    del moyenne_y, classes, yi_classe, SCT, SCE
    gc.collect()
    return eta_squared


def welch_ttest(x, y, alpha=0.05):
    """
    Test de Welch avec H0: égalité des moyennes entre x et y.
    :param x: numpy array ou pandas Series
    :param y: numpy array ou pandas Series
    :param alpha:seuil de test de la p-value ; default=0.05
    :return: bool:
        - True: H0 vraie (égalité)
        - False: H0 rejetée (inégalité)
    """
    dof = (x.var() / x.size + y.var() / y.size) ** 2 / (
            (x.var() / x.size) ** 2 / (x.size - 1) + (y.var() / y.size) ** 2 / (y.size - 1))
    t, p = st.ttest_ind(x, y, equal_var=False)
    result = p > alpha
    del dof, t, p
    gc.collect()
    return result


from numpy.polynomial import polynomial as P
import warnings
def anova(df_cat_num, nb_cat=5, alpha=0.05, save=None, verbose=0):
    """
    Effectue l'ANOVA pour une paire de variables (cat, num).
    :param data: dataframe, contenant en ligne toutes les
        observations et 2 colonnes, une pour chaque feature.
    :param nb_cat: int, nombre maximum de catégories à afficher
        pour la variables catégorielle.
    :param alpha: float, seuil des tests (normalité, Welch, Fligner).
    :param save: str, chemin vers le fichier d'enregistrement du
        graphique ; default=None, pas d'enregistrement.
    :param verbose: int, niveau de verbosité:
        - 0: n'affiche que le graphique et filtre les UserWarning
        - 1: affiche de plus les résultats des tests et la régression
            linéaire.
        - 2 ou plus: affiche de plus les informations sur les
            résultats de test et les 'UserWarning'.
    :return:
    """
    pair = df_cat_num.columns.tolist()
    if len(pair) != 2:
        print(f"Le dataframe ne correspond pas à une paire de features.")
        return
    if verbose>0:
        print(Fore.GREEN + "► ANOVA pour la paire : " + Style.RESET_ALL, pair)
    df = df_cat_num[pair].copy(deep=True)

    # Filtrage des UserWarning en mode non verbose
    if verbose < 2:
        warnings.filterwarnings(action="ignore", category=UserWarning)

    # Format des étiquettes de valeur unique
    cat_feat = df.columns.tolist()[0]
    is_feat_num = True if np.issubdtype(df[cat_feat].dtype, np.number) else False
    if is_feat_num:
        df[cat_feat] = pd.to_numeric(df[cat_feat], errors='coerce')
        is_int = np.array([x % 1 == 0 for x in pd.unique(df[cat_feat])]).all()
        if is_int:
            df[cat_feat] = df[cat_feat].astype(int)

    # Filtrage des catégories qui contiennent moins de 'n_samples_per_cat_min' lignes (min=3)
    df_cat = df.groupby(pair[0], as_index=False).agg(
        means=(pair[1], "mean"),size=(pair[0], "size")).sort_values(
        by='means', ascending=False).reset_index(drop=True)
    n_samples_per_cat_min = 3
    list_cat = df_cat.loc[df_cat['size'] >= n_samples_per_cat_min, pair[0]].tolist()
    df.drop(index=df.loc[~df[pair[0]].isin(list_cat), :].index, inplace=True)
    df_cat.drop(index=df_cat.loc[~df_cat[pair[0]].isin(list_cat), :].index, inplace=True)
    df_cat.reset_index(drop=True, inplace=True)

    # Filtrage des nb_cat pour lesquelles la moyenne des valeurs numériques est la plus élevée
    df_cat = df_cat.head(nb_cat)
    list_cat = df_cat[pair[0]].head(nb_cat).values.tolist()
    nb_cat = min(nb_cat, len(list_cat))
    df.drop(index=df.loc[~df[pair[0]].isin(list_cat), :].index, inplace=True)
    df[pair[0]] = pd.Categorical(df[pair[0]], categories=list_cat, ordered=True)

    # Calcul du rapport de corrélation
    eta_sqr = eta_squared(df[pair[0]], df[pair[1]])
    if verbose > 0:
        print("  → Rapport de corrélation pour les k=", nb_cat, "catégories du graphique et n=",
              df.shape[0], "données : η²=" + f"{eta_sqr:.3f}")

    # Remplacement des catégories par une valeur numérique
    df['cat'] = df[pair[0]].copy()
    df['cat'] = df['cat'].astype('object').astype("category")
    df['cat'].replace(df['cat'].cat.categories, [i for i in range(0, len(df['cat'].cat.categories))], inplace=True)
    df['cat'] = df['cat'].astype("int")

    # Tests sur les variables
    # Test de normalité (H0: distribution normale)
    tn = True
    list_norm_neg = {'category': [], 'statistic': [], 'p-value': []}
    for cat in range(nb_cat):
        stat, pvalue = st.normaltest(df.loc[df['cat'] == cat, pair[1]].values)
        tn = tn and (pvalue > alpha)
        if pvalue <= alpha:
            list_norm_neg['category'].append(cat)
            list_norm_neg['statistic'].append(stat)
            list_norm_neg['p-value'].append(pvalue)
    if verbose>0:
        if tn:
            print("  → Test de normalité positif pour toutes les catégories")
        else:
            print("  → Test de normalité négatif sur certaines catégories")
            if verbose>1:
                display(pd.DataFrame.from_dict(list_norm_neg))

    # Test d'homoscédasticité (H0: variances égales entre les catégories)
    gb = df.groupby(pair[0])[pair[1]]
    stat, p_fligner = st.fligner(*[gb.get_group(x).values for x in gb.groups.keys()])
    is_fligner_test_positive = p_fligner > alpha
    if verbose > 0:
        if is_fligner_test_positive:
            print("  → Test d'homoscédasticité de Fligner-Killeen positif "
                  "(Ecarts types égaux entre les catégories)")
        else:
            print(f"  → Test d'homoscédasticité de Fligner-Killeen négatif "
                  f"(Ecarts types non égaux entre les catégories)")
            if verbose > 1:
                std = pd.DataFrame(data=[gb.get_group(x).values.std() for x in gb.groups.keys()],
                                   columns=['std'], index=gb.groups.keys())
                display(std)

    # Test de Welch (H0: égalité des moyennes entre catégories), si test d'homoscédasticité négatif
    # Table de groupe des catégories en fonction du résultat du test
    tw_true = True
    tw_false = True
    dgr = pd.DataFrame(data=np.arange(len(list_cat)), index=[list_cat], columns=['group'])
    for i in range(len(list_cat) - 1):
        for j in range(i + 1, len(list_cat)):
            is_welch_ttest_positive = welch_ttest(gb.get_group(list_cat[i]).values, gb.get_group(list_cat[j]).values)
            tw_true = tw_true and is_welch_ttest_positive
            tw_false = tw_false and not is_welch_ttest_positive
            # Si le test est positif, les moyennes des 2 catégories sont équivalentes
            if is_welch_ttest_positive:
                gr = dgr.loc[list_cat[i]]['group']
                dgr.at[list_cat[j], 'group'] = gr
    # Valeurs de l'ordonnée pour le grouper les catégories ayant des moyennes non dissemblables
    rows = [-0.5]
    for i in range(1, len(list_cat)):
        if dgr['group'].values[i]!=dgr['group'].values[i-1]:
            rows.append(i-0.5)
    rows.append(len(list_cat)-0.5)
    # Affichage du résultat du test de Welch
    if verbose > 0:
        if tw_true:
            print("  → Test de Welch positif (égalité des moyennes) "
                  "pour toutes les catégories")
        elif tw_false:
            print("  → Test de Welch (égalité des moyennes) négatif "
                  "pour toutes les catégories")
        else:
            print("  → Test de Welch (égalité des moyennes) positifs "
                  "pour les catégories de même groupe sur le graphique")

    # Test statistique de Fisher
    dfn = nb_cat - 1
    dfd = df.shape[0] - nb_cat
    F_crit = st.f.ppf(1 - alpha, dfn, dfd)
    F_stat, p = st.f_oneway(df['cat'], df[pair[1]])
    sign_F = ">" if F_stat > F_crit else "<"
    sign_p = ">" if p > alpha else "<"
    if (sign_F == ">") and (sign_p == "<"):
        res_test = "positif"
    else:
        res_test = "négatif"
    if verbose > 0:
        print(f"  → Test de Fisher {res_test}")
        if verbose>1:
            print(f"\tF={F_stat:.2f} {sign_F} {F_crit:.2f}",
              f" , et p-value={p:.2e} {sign_p} {alpha:0.2f}")

    # Définition des dimensions du graphique global
    fig_h = nb_cat if nb_cat < 6 else int((5 * nb_cat + 40) / 15)

    # Propriétés graphiques
    medianprops = {'color': "black"}
    meanprops = {'marker': 'o', 'markeredgecolor': 'black', 'markerfacecolor': 'firebrick'}

    fig, ax = plt.subplots(figsize=(15, fig_h))
    ax = sns.boxplot(x=pair[1], y=pair[0], data=df, showfliers=False, ax=ax,
                     medianprops=medianprops, showmeans=True, meanprops=meanprops)
    xmin, xmax = ax.get_xlim()

    # Tracé des lignes reliant les valeurs moyennes de chaque catégorie
    plt.plot(df_cat.means.values, df_cat.index.values, linestyle='--', c='#000000')

    # Bloc de séparation graphique des groupes de moyennes non différenciées (test de Welch négatif)
    if not tw_true and len(rows)>1:
        for i in range(len(rows)-1):
            plt.fill_between([xmin, xmax], [rows[i], rows[i]], [rows[i+1], rows[i+1]], alpha=0.2)

    # Régression linéaire sur les valeurs moyennes
    reg = P.polyfit(df_cat.means.values, df_cat.index.values, deg=1, full=True)
    yPredict = P.polyval(df_cat.means.values, reg[0])

    if verbose > 0:
        if nb_cat > 2:
            coef_cor = 1 - reg[1][0][0] / (np.var(df_cat.index.values) * len(df_cat.index.values))
        else:
            coef_cor = 1
        a = -1 / reg[0][1]
        mu = -reg[0][0] / reg[0][1] - a * (df_cat.shape[0] - 1)
        sign = '+' if a >= 0 else '-'
        print(f"\n  → Moyenne catégorielle : '{pair[1]}' = {mu:.2f}  {sign} {abs(a):.2f} * '{pair[0]}', avec :",
              f"'{df_cat[pair[0]][df_cat.shape[0] - 1]}'= 0 , …, '{df_cat[pair[0]][0]}'= {df_cat.shape[0] - 1}")
        print(f"  → Coefficient de corrélation r² ={coef_cor:.2f}")

    # Tracé de la droite de régression linéaire
    plt.plot(df_cat.means.values, yPredict, linewidth=2, linestyle='-', c='#FF0000')

    plt.ylim(top=-1, bottom=nb_cat)
    plt.title(f"ANOVA - analyse bivariée (corrélation η²={eta_sqr:.3f})", fontsize=14)
    plt.tight_layout()
    if save is not None:
        plt.savefig(save, dpi=300)
    plt.show()

    # Retour à la gestion des warnings par défaut
    if verbose > 1:
        warnings.filterwarnings(action="default", category=UserWarning)

    # Nettoyage des variables
    del pair, df, df_cat, n_samples_per_cat_min, list_cat
    del nb_cat, eta_sqr, tn, list_norm_neg, cat, stat, pvalue
    del gb, p_fligner, is_fligner_test_positive, tw_true
    del tw_false, dgr, i, j, p, is_welch_ttest_positive
    del rows, dfn, dfd, F_crit, F_stat, sign_F, sign_p
    del res_test, fig_h, medianprops, meanprops, ax
    del xmin, xmax, reg, yPredict, cat_feat, is_feat_num
    if 'gr' in locals(): del gr
    if 'coef_cor' in locals(): del coef_cor
    if 'a' in locals(): del a
    if 'mu' in locals(): del mu
    if 'is_int' in locals(): del is_int
    gc.collect()


def pair_plot(data, save=None):
    pair = data.columns.tolist()
    if len(pair) != 2:
        print(f"Le dataframe ne correspond pas à une paire de features.")
        return
    df = data[pair].copy().apply(pd.to_numeric, axis=1)
    coef_p = st.pearsonr(df[pair[0]], df[pair[1]])[0]
    plot = sns.jointplot(data=df, x=pair[0], y=pair[1], kind="reg", marginal_kws=dict(bins=20, fill=True))
    plt.suptitle(f"Analyse bivariée (corrélation r²={coef_p:.3f})", fontsize=14)
    plt.tight_layout()
    if save is not None:
        plt.savefig(save, dpi=300)
    plt.show()
    del pair, df, coef_p, plot
    gc.collect()

