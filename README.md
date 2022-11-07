# OCP7-scoring-model-implementation
Projet 7 d'Openclassrooms / CentraleSupélec

Le projet consiste à mettre en oeuvre un modèle de "scoring crédit" s'appuyant sur des sources de données variées relatives à la demande de crédit et aux antécédants de crédit des clients. L'analyse des données brutes est représentée sur un graphe avec networkx pour montrer les relations entre les tables de données et en déduire le processus d'assemblage approprié.

Le feature engineering est issu de l'analyse d'un compétiteur Kaggle (Aguiar) pour déboucher sur la création de 10 features supplémentaires.

Le dataset résultant de l'assemblage et du feature engineering a pour dimensions 356251 lignes et 797 colonnes.
Compte tenu du nombre élevé de features, le préprocessing (traitement des valeurs manquantes, traitement du skew/kurtosis et mise à l'échelle) est effectué de manière automatisée avec possibilité de forçage concernant les valeurs manquantes.

Le modèle de classification identifie la probabilité de risque de crédit client à partir d'un jeu déséquilibré: classification binaire selon le risque de non remboursement, avec probabilité de classe associée.

La recherche du meilleur modèle débouche sur la sélection d'un LGBMClassifier dont les hyperparmètres sont optimisés avec optuna, à la fois sous l'angle technique et celui du métier, en utilisant les métriques appropriées (PR AUC et fbeta score).

Le modèle est présenté sous la forme d'un dashboard, dont la partie calcul est réalisée par une API (fastAPI), et la partie visualisation par Streamlit. L'interprétation du modèle s'effectue sur la base des valeurs de Shapley avec la librairie Shap.
Cette interprétation inclut l'analyse bivariée de 2 features à sélectionner, analyse différenciée selon que chaque variable est numérique ou catégorielle (3 cas).
Le dictionnaire des features est fourni dans un onglet du dashboard.

L'API (à lancer en premier: temps de chargement 30s à 1mn) est déployée sur le web avec heroku à l'adresse: https://ocp7-dbbackend.herokuapp.com/docs.
Le dashboard graphique (à lancer une fois l'API en fonctionnement) est déployé sur le web avec heroku à l'adresse: https://ocp7-dbfrontend.herokuapp.com.
