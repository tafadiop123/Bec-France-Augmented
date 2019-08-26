# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 02:47:59 2019

@author: Moustapha
"""

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
warnings.filterwarnings('ignore')
import datetime as dt
from collections import Counter
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score


bec = pd.read_csv("becfrance.csv", sep = ';', parse_dates=[10,11,24], dayfirst = True, engine='python', encoding=None)
#bec2 = pd.read_csv("becfrance2.csv", sep = ';', parse_dates=[10,11,24], dayfirst = True, engine='python', encoding=None)

#Comprehension de la base 
bec.info()
bec.describe()
bec.hist(bins=50, figsize=(20,15))
plt.show()
bec['ETABLISSEMENT'].value_counts()

#Déterminons un dictionnaire qui représente les catégories d'établissement
categorie={
    "COLLEGE" : "COLLEGE",
    "LEAP" : "LYCEE",
    "LP" : "LYCEE",
    "LYCEE" : "LYCEE",
}

# Assignation des différents établissements à une categorie 
def detect_words(values, dictionary):
    result = []
    for lib in values:
        operation_type = "AUTRES"
        for word, val in dictionary.items():
            if word in lib:
                operation_type = val
        result.append(operation_type)
    return result
bec["categ_etab"] = detect_words(bec["ETABLISSEMENT"], categorie)

bec['categ_etab'].value_counts()

bec["categ_etab"].value_counts(normalize=True).plot(kind='bar')
plt.show()

bec['STATUTDEVIS'].value_counts()

# Création de la variable target binarisée
def target(series):
    if series == 'Devis rejeté':
        return 'n_signe'
    elif series == 'Première demande':
        return 'n_signe'
    elif series == 'Contrat envoyé':
        return 'n_signe'
    elif series == 'Deuxième demande':
        return 'n_signe'
    elif series == 'Client intéressé':
        return 'n_signe'
    elif series == 'Contrat réceptionné signé':
        return 'signe'
    elif series == 'Confirmation formelle':
        return 'signe'
    elif series == 'Confirmation nonformelle':
        return 'signe'

bec['devis target'] = bec['STATUTDEVIS'].apply(target)
bec['devis target'].value_counts()

# Définition d'une fonction générique en utilisant la fonction replace de Pandas
def coding(col, codeDict):
    colCoded = pd.Series(col, copy=True)
    for key, value in codeDict.items():
        colCoded.replace(key, value, inplace=True)
    return colCoded

#recodage de la variable cible numériquement
bec["devis target coded"] = coding(bec["devis target"], {'n_signe':0,'signe':1})

bec = bec.fillna(0)       
#######################################################################################
## D'aprés le coefficient de correlation on remarque qu'il existe une correlation intéressante avec le (kilometrageAllerRetourTotalAutocarRetenu = -0,24)
## et une correlation infime avec le (montantTotal = 0,0495) 
#######################################################################################

# Détermination de la distribution de la variable target
#""""""""##
def add_freq():
    ncount = len(bec)

    ax2=ax.twinx()

    ax2.yaxis.tick_left()
    ax.yaxis.tick_right()

    ax.yaxis.set_label_position('right')
    ax2.yaxis.set_label_position('left')

    ax2.set_ylabel('Frequency')

    for p in ax.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
                ha='center', va='bottom')

    ax2.set_ylim(0,100)
    ax2.grid(None)
    
####""""""########
ax = sns.countplot(x = bec["devis target coded"] ,palette="Set3") 
sns.set(font_scale=1.5)
ax.set_xlabel('devis signé ou non')
ax.set_ylabel('Frequency')
fig = plt.gcf()
fig.set_size_inches(10,5)
add_freq()
plt.show()

#################################################################################
## Nous remarquons que notre variable target est trés asymétrique c'est-à-que notre variable cible est déséquimibré (ou imbalanced)
###################################################################################"


# Calcul pour les dates 

bec['dateDebutSejour'] = pd.to_datetime(bec['dateDebutSejour'])
bec['dateFinSejour'] = pd.to_datetime(bec['dateFinSejour'])
bec['DATEDEVIS'] = pd.to_datetime(bec['DATEDEVIS'])

## Spécification de l'année et du mois de la commande de devis
bec["annee"] = bec["DATEDEVIS"].map(lambda d: d.year)
bec["mois"] = bec["DATEDEVIS"].map(lambda d: d.month)

bec["duree sejour"] = bec["dateFinSejour"]-bec["dateDebutSejour"]
bec['duree sejour'] = pd.to_numeric(bec['duree sejour'], errors='ignore')

bec_fr = bec.copy() # pour copier le dataframe afin qu'il soit indépendant

#La matrice de correlation
corr = pd.DataFrame()
corr = bec_fr.corr()

# Faisons la représentation graphique de la distribution empirique de certaines de nos variables
# en fonction de la variable target

bec0 = bec[bec["devis target coded"] == 0 ]
bec1 = bec[bec["devis target coded"] == 1 ]

## Diagramme cammambert des pays de destinantions non signées

bec0["PAYS"].value_counts(normalize=True).plot(kind='pie')
plt.title("les destinations non signées")
plt.axis('equal')  # Cette ligne assure que le pie chart est un cercle plutôt qu'une éllipse
plt.show() 

## Diagramme cammambert des pays de destinantions signées

bec1["PAYS"].value_counts(normalize=True).plot(kind='pie')
plt.title("les destinations signées")
plt.axis('equal')  # Cette ligne assure que le pie chart est un cercle plutôt qu'une éllipse
plt.show() 

## Diagramme en barres pour voir les devis signées en fontions des reductions accordées 

bec1["Reduction"].value_counts(normalize=True).plot(kind='bar')
plt.title("réductions - signées")
plt.show()

## Diagramme en barres pour voir les devis signées en fontions des reductions accordées 

bec0["Reduction"].value_counts(normalize=True).plot(kind='bar')
plt.title("réductions -  non signées")
plt.show()

## Diagramme pour voire quel est le type de transport le plus choisi par les devis signés
bec1["typeTransport"].value_counts(normalize=True).plot(kind='barh')
plt.title("transport - devis signées")
plt.show()
## Diagramme pour voire quel est le type de transport le plus choisi par les devis non signés
bec0["typeTransport"].value_counts(normalize=True).plot(kind='barh')
plt.title("transport - devis non signées")
plt.show()

## Diagramme des catégories d'établissements qui ont le plus signées 
bec1["categ_etab"].value_counts(normalize=True).plot(kind='pie', autopct='%.2f')
plt.title("établissements - devis signées")
plt.show()

## Diagramme des catégories d'établissements qui ont le plus signées 
bec0["categ_etab"].value_counts(normalize=True).plot(kind='pie', autopct='%.2f')
plt.title("établissements - devis non signées")
plt.show()


####### Faisons une représentation des devis non signées et signées durant les années et les mois #######
bec0["valeur"] = 1
bec_time = bec0.groupby(['mois','devis target coded' ,'annee'])['valeur'].sum().reset_index()
bectime2 = bec_time.pivot(index='annee', columns='mois', values='valeur')
bectime2 = bectime2.fillna(0)



bec1["valeur"] = 1
bec_time_s = bec1.groupby(['mois','devis target coded' ,'annee'])['valeur'].sum().reset_index()
bectime3 = bec_time_s.pivot(index='annee', columns='mois', values='valeur')
bectime3 = bectime3.fillna(0)

################ Faisons une détection des valeurs abérentes ##############


def detect_outliers(df,n,features):
    outlier_indices = []
    
    # Itération sur les collones
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # limite des outliers
        outlier_step = 1.5 * IQR
        
        # Détermination d'une liste d'indices des outliers pour la fonction col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # ajouter les indices des outliers trouvées pour col à la liste des indices des outliers
        outlier_indices.extend(outlier_list_col)
        
    # Sélection des observations qui contiennent plus de 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers

# Supression des outliers que l'on retrouve dans le dataset
Outliers_to_drop = detect_outliers(bec,2,["nombreElevesMoins16",
                                            "nombreElevesEntre16et18",
                                            "nombreEleves18etPlus",
                                            "nombreAccompagnant",
                                            "montantTotal",
                                            "kilometrageAllerRetourTotalAutocarRetenu"])

bec.loc[Outliers_to_drop]

###################################################################################################
########### MISE EN PLACE DE L'ALGORITHME DE PREDICTION ###########################################
###################################################################################################

## Notre base se compose de variable qualitatives et quantitatives donc nous allons utiliser 
## la fonction labelEncoder pour encoder les variables catégorielles
bec['kilometrageAllerRetourTotalAutocarRetenu'] = bec['kilometrageAllerRetourTotalAutocarRetenu'].astype('int64')
bec['montantTotal'] = bec['montantTotal'].astype('int64')

bec.drop(labels=["small", "VILLEETAB", "NUMERODEVIS", "DESTINATION", "dateDebutSejour", "dateFinSejour", "EMAILCREATEUR", "DATEDEVIS", "devis target" ],axis = 1,inplace=True)


labelEncoder = preprocessing.LabelEncoder()

bec['ETABLISSEMENT']=labelEncoder.fit_transform(bec['ETABLISSEMENT'])
bec['ACADEMIE']=labelEncoder.fit_transform(bec['ACADEMIE'])
bec['PAYS']=labelEncoder.fit_transform(bec['PAYS'])
bec['Reduction']=labelEncoder.fit_transform(bec['Reduction'])
bec['typeTransport']=labelEncoder.fit_transform(bec['typeTransport'])
bec['categ_etab']=labelEncoder.fit_transform(bec['categ_etab'])

#Les variables explicatives
X = bec[['email','ETABLISSEMENT','CPETAB', 'ACADEMIE', 'PAYS', 'ETATSCENARIO',
         'nombreElevesMoins16', 'nombreElevesEntre16et18', 'nombreEleves18etPlus','nombreAccompagnant',
         'montantTotal', 'Reduction', 'kilometrageAllerRetourTotalAutocarRetenu', 'typeTransport',
         'ETATDEVIS', 'Appartenance', 'STATUTDEVIS', 'categ_etab', 'devis target coded',
         'devis target coded', 'annee', 'mois', 'duree sejour']]

#La variable target
Y = bec['devis target coded']

### Divisons de la base en train et test

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

### Copie des 2 parties de la base test
X_test_result = X_test.copy()
Y_test_result = Y_test.copy()

### Suppression des variables inutilisables 
X_train.drop(labels=["email", "ETATDEVIS", "STATUTDEVIS", "ETATSCENARIO", "Appartenance"],axis = 1,inplace=True)
X_test.drop(labels=["email", "ETATDEVIS", "STATUTDEVIS", "ETATSCENARIO", "Appartenance"],axis = 1,inplace=True)
X_train.drop(labels=["devis target coded"],axis = 1,inplace=True)
X_test.drop(labels=["devis target coded"],axis = 1,inplace=True)

X_train.info()       
    
## Faisons une normalisation et une standarisation du dataset

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
scaler1 = preprocessing.StandardScaler().fit(X_test)
X_test = scaler.transform(X_test)  

### Déterminons le modéle d'apprentissage avec GradientBoostingClassifier 
type_classifier = GradientBoostingClassifier
gbc_bec = type_classifier()
gbc_bec = gbc_bec.fit(X_train, Y_train.ravel())   ### ravel permet d'éviter de prendre en compte l'index de Y_train 

##########
### Détermination des résultats du modèle sur la base de test

for x,y in [ (X_train, Y_train), (X_test, Y_test) ]:
    yp  = gbc_bec.predict(x)
    conf_mat = confusion_matrix(y.ravel(), yp.ravel())      
    print(conf_mat)
plt.matshow(conf_mat)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')

### Regardons l'importance de nos variables dans le modèle
#X.drop(labels=["email", "ETATDEVIS", "STATUTDEVIS", "ETATSCENARIO","Appartenance"],axis = 1,inplace=True)

feature_name = X.columns
limit = 15
feature_importance = gbc_bec.feature_importances_[:15]
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, feature_name[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')

###### d'aprés le plot sur sur l'importance des variables on en déduis que les variables qui participent à la construction de notre modèle sont:
## "kilométrage du voyage", "mois", "CPETAB", "montantTotal", "nombreElevesmoins16", "annee"


###### Représentation de la courbe ROC pour évaluer le score du model

probas = gbc_bec.predict_proba(X_test)
probas[:]

rep = [ ]
yt = Y_test.ravel()
for i in range(probas.shape[0]):
    p0,p1 = probas[i,:]
    exp = yt[i]
    if p0 > p1 :
        if exp == 0 :
            # bonne réponse
            rep.append ( (1, p0) )
        else :
            # mauvaise réponse
            rep.append( (0,p0) )
    else :
        if exp == 0 :
            # mauvaise réponse
            rep.append ( (0, p1) )
        else :
            # bonne réponse
            rep.append( (1,p1) )
mat_rep = np.array(rep)
mat_rep[:]

fpr, tpr, thresholds = roc_curve(mat_rep[:,0], mat_rep[:, 1])
roc_auc = auc(fpr, tpr)
plt_roc = plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt_roc_plot = plt.plot([0, 1], [0, 1], 'k--')
plt_roc_xlim = plt.xlim([0.0, 1.0])
plt_roc_ylim = plt.ylim([0.0, 1.0])
plt_roc_xlab = plt.xlabel('False Positive Rate')
plt_roc_ylabel = plt.ylabel('True Positive Rate')
plt_roc_title = plt.title('ROC')
plt_roc_legend = plt.legend(loc="lower right")

score_model = "taux de bonne réponse",sum(mat_rep[:,0]/len(mat_rep))

### La matrice de confusion
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements 
accuracy(conf_mat)

### Mettre les scores sur un dataframe
score = pd.DataFrame(probas)
score_signe = score [ 1 ]
score_signe = score_signe.to_frame()
score_signe = score_signe*100

## export des bases Xtest result et score signe sous excel
X_test_result.to_csv('test_result.csv', sep=',')
score_signe.to_csv('score_signe.csv', sep=',')

## Importation de la base finale
prospect_scoring = pd.read_csv("prospect_scoring.csv", sep = ';', engine='python', encoding=None)

#### Cross validation pour evaluer la precison du score du modele
score_cross_val = cross_val_score(gbc_bec, X_test, Y_test, cv=10)
score_cross_val             
### On remarque que nos résultat sont proches du score retenu ####   



                             