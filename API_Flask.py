from joblib import load
from flask import Flask, jsonify, request, jsonify, render_template
import json
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling  import RandomUnderSampler
import xgboost as xgb
from lightgbm import LGBMClassifier as lgbm



# Create an instance of the Flask class that is the WSGI application.
# The first argument is the name of the application module or package,
# typically __name__ when using a single module.
app = Flask(__name__)


# On charge les données
data_train = pd.read_csv("Codes/data/application_train.csv")
data_test = pd.read_csv("Codes/data/application_test.csv")

# On charge le jeu d'entrainement et le modèle optimal
X_train = pickle.load(open('Codes/X_train_mediane.sav', 'rb'))
X_test = pickle.load(open('Codes/X_test_mediane.sav', 'rb'))

url_df = 'Codes/data_mediane.csv'
df_med = pd.read_csv(url_df)
df_med['SK_ID_CURR'] = df_med['SK_ID_CURR'].astype('int')

liste_clients = list(df_med['SK_ID_CURR'].sort_values())


LGBM_final = pickle.load(open('Codes/modele_lgbm_final.sav', 'rb'))

train = None
test = None
model = None

# On crée la liste des ID clients qui nous servira dans l'API
id_client = df_med["SK_ID_CURR"][:50].values
id_client = pd.DataFrame(id_client)

# routes
# Entraînement du modèle
@app.route("/init_model", methods=["GET"])
def init_model():
    
    # On prépare les données
    df_train, df_test = features_engineering(data_train, data_test)

    print("Features engineering done")
    
    # On fait le préprocessing des données
    df_train, df_test = preprocesseur(df_train, df_test)

    # On transforme le dataset de test préparé en variabe
    # globale, car il est utilisé dans la fonction predict
    global train
    train = df_train.copy()

    global test
    test = df_test.copy()

    print("Preprocessing done")
    
    # On fait un resampling des données d'entraînement
    X, y = data_resampler(df_train, test)
    print("Resampling done")

    # On entraîne le modèle et on le transforme en
    # variable globale pour la fonction predict
    global clf_lgbm
    clf_lgbm = entrainement_lgbm(X, y)
    print("Training lgbm done")

    global knn
    knn = entrainement_knn(df_train)
    print("Training knn done")

    return jsonify(["Initialisation terminee."])


@app.route('/<int:ID_client>', methods=["POST", "GET"])
def predict_client(ID_client):
    seuil = 0.66

    if ID_client in liste_clients:
        X = df_med[df_med['SK_ID_CURR'] == ID_client]
        X.drop('SK_ID_CURR', axis=1, inplace=True)

        predictions_proba = LGBM_final.predict_proba(X)
        predict_proba_0 = round(predictions_proba[0, 0], 3)
        predict_proba_1 = round(predictions_proba[0, 1], 3)

        if (1-predictions_proba[:, 1]) >= seuil:
            predict = 'Oui'
        elif (1-predictions_proba[:, 1]) < seuil:
            predict = 'Non'

         
    else:
        predict = 'Client inconnu'
        predict_proba_0 = 'Client inconnu'
        predict_proba_1 = 'Client inconnu'

    return jsonify({ 'retour_prediction' : str(predict),
                     'predict_proba_0': str(1-predict_proba_0),
                     'predict_proba_1': str(1-predict_proba_1) })
##########################################################################################

@app.route('/accord/<int:ID_client>', methods=["POST", "GET"])
def predict_client_accord(ID_client):
    seuil = 0.66

    if ID_client in liste_clients:
        X = df_med[df_med['SK_ID_CURR'] == ID_client]
        X.drop('SK_ID_CURR', axis=1, inplace=True)

        predictions_proba = LGBM_final.predict_proba(X)
        predict_proba_0 = round(predictions_proba[0, 0], 3)
        predict_proba_1 = round(predictions_proba[0, 1], 3)

        if (1-predictions_proba[:, 1]) >= seuil:
            predict = 'Oui'
        elif (1-predictions_proba[:, 1]) < seuil:
            predict = 'Non'

         
    else:
        predict = 'Client inconnu'
        predict_proba_0 = 'Client inconnu'
        predict_proba_1 = 'Client inconnu'

    return jsonify(str(predict) )

############################################################
@app.route('/score/<int:ID_client>', methods=["POST", "GET"])
def predict_client_score(ID_client):
    seuil = 0.66

    if ID_client in liste_clients:
        X = df_med[df_med['SK_ID_CURR'] == ID_client]
        X.drop('SK_ID_CURR', axis=1, inplace=True)

        predictions_proba = LGBM_final.predict_proba(X)
        predict_proba_0 = round(predictions_proba[0, 0], 3)
        predict_proba_1 = round(predictions_proba[0, 1], 3)
         
    else:
        predict = 'Client inconnu'
        predict_proba_0 = 'Client inconnu'
        predict_proba_1 = 'Client inconnu'

    return jsonify(str(1-predictions_proba[:,1][0]) )


# Chargement des données pour la selection de l'ID client
@app.route("/load_data", methods=["GET"])
def load_data():
    
    return id_client.to_json(orient='values')

# Chargement d'informations générales
@app.route("/infos_gen", methods=["GET"])
def infos_gen():

    lst_infos = [data_train.shape[0],
                 round(data_train["AMT_INCOME_TOTAL"].mean(), 2),
                 round(data_train["AMT_CREDIT"].mean(), 2)]

    return jsonify(lst_infos)

# Chargement des données pour le graphique
# dans la sidebar
@app.route("/disparite_target", methods=["GET"])
def disparite_target():

    df_target = data_train["TARGET"].value_counts()

    return df_target.to_json(orient='values')

# Chargement d'informations générales sur le client
@app.route("/infos_client", methods=["GET"])
def infos_client():

    id = request.args.get("id_client")

    data_client = data_test[data_test["SK_ID_CURR"] == id]

    print(data_client)
    
    dict_infos = {
#        "status_famille" : data_client["NAME_FAMILY_STATUS"].tolist(),
#        "nb_enfant" : data_client["CNT_CHILDREN"].tolist(),
# #        "age" : int(data_client["DAYS_BIRTH"].values / -365),
#        "revenus" : data_client["AMT_INCOME_TOTAL"].tolist(),
#        "montant_credit" : data_client["AMT_CREDIT"].tolist(),
#        "annuites" : data_client["AMT_ANNUITY"].tolist(),
#        "montant_bien" : data_client["AMT_GOODS_PRICE"].tolist()
       }
    
    response = json.loads(data_client.to_json(orient='index'))

    return response

# Calcul des ages de la population pour le graphique
# situant l'age du client
@app.route("/load_age_population", methods=["GET"])
def load_age_population():
    
    df_age = round((data_train["DAYS_BIRTH"] / -365), 2)
    return df_age.to_json(orient='values')

# Segmentation des revenus de la population pour le graphique
# situant l'age du client
@app.route("/load_revenus_population", methods=["GET"])
def load_revenus_population():
    
    # On supprime les outliers qui faussent le graphique de sortie
    # Cette opération supprime un peu moins de 300 lignes sur une
    # population > 300000...
    df_revenus = data_train[data_train["AMT_INCOME_TOTAL"] < 700000]
    
    df_revenus["tranches_revenus"] = pd.cut(df_revenus["AMT_INCOME_TOTAL"], bins=20)
    df_revenus = df_revenus[["AMT_INCOME_TOTAL", "tranches_revenus"]]
    df_revenus.sort_values(by="AMT_INCOME_TOTAL", inplace=True)

    print(df_revenus)
    
    df_revenus = df_revenus["AMT_INCOME_TOTAL"]

    return df_revenus.to_json(orient='values')

@app.route("/predict", methods=["GET"])
def predict():
    
    id = request.args.get("id_client")

    print("Analyse data_test :")
    print(data_test.shape)
    print(data_test[data_test["SK_ID_CURR"] == id]) #int(id)
     
    index = data_test[data_test["SK_ID_CURR"] == id].index.values

    print(index[0])
   
    data_client = test[index]

    print(data_client)

    prediction = clf_lgbm.predict_proba(data_client)

    prediction = prediction[0].tolist()

    print(prediction)

    return jsonify(prediction)

@app.route("/load_voisins", methods=["GET"])
def load_voisins():
    
    id = request.args.get("id_client")

    index = data_test[data_test["SK_ID_CURR"] == int(id)].index.values

    data_client = test[index]
    
    distances, indices = knn.kneighbors(data_client)

    print("indices")
    print(indices)
    print("distances")
    print(distances)

    df_voisins = data_train.iloc[indices[0], :]
    
    response = json.loads(df_voisins.to_json(orient='index'))

    return response


def features_engineering(data_train, data_test):

    # Cette fonction regroupe toutes les opérations de features engineering
    # mises en place sur les sets train & test

    #############################################
    # LABEL ENCODING
    #############################################
    # Create a label encoder object
    le = LabelEncoder()
    le_count = 0

    # Iterate through the columns
    for col in data_train.columns:
        if data_train[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(data_train[col].unique())) <= 2:
                # Train on the training data
                le.fit(data_train[col])
                # Transform both training and testing data
                data_train[col] = le.transform(data_train[col])
                data_test[col] = le.transform(data_test[col])
                
                # Keep track of how many columns were label encoded
                le_count += 1

    ############################################
    # ONE HOT ENCODING
    ############################################
    # one-hot encoding of categorical variables
    data_train = pd.get_dummies(data_train)
    data_test = pd.get_dummies(data_test)

    train_labels = data_train['TARGET']
    # Align the training and testing data, keep only columns present in both dataframes
    data_train, data_test = data_train.align(data_test, join = 'inner', axis = 1)
    # Add the target back in
    data_train['TARGET'] = train_labels

    ############################################
    # VALEURS ABERRANTES
    ############################################
    # Create an anomalous flag column
    data_train['DAYS_EMPLOYED_ANOM'] = data_train["DAYS_EMPLOYED"] == 365243
    # Replace the anomalous values with nan
    data_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
    data_test['DAYS_EMPLOYED_ANOM'] = data_test["DAYS_EMPLOYED"] == 365243
    data_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

    # Traitement des valeurs négatives
    data_train['DAYS_BIRTH'] = abs(data_train['DAYS_BIRTH'])

    ############################################
    # CREATION DE VARIABLES

    # In this frame of mind, we can make a couple features that attempt to capture what we think 
    # may be important for telling whether a client will default on a loan. 
    # Here I'm going to use five features that were inspired by this script by Aguiar:

    # CREDIT_INCOME_PERCENT: the percentage of the credit amount relative to a client's income
    # ANNUITY_INCOME_PERCENT: the percentage of the loan annuity relative to a client's income
    # CREDIT_TERM: the length of the payment in months (since the annuity is the monthly amount due
    # DAYS_EMPLOYED_PERCENT: the percentage of the days employed relative to the client's age
    # Again, thanks to Aguiar and his great script for exploring these features.
    
    data_train_domain = data_train.copy()
    data_test_domain = data_test.copy()

    data_train_domain['CREDIT_INCOME_PERCENT'] = data_train_domain['AMT_CREDIT'] / data_train_domain['AMT_INCOME_TOTAL']
    data_train_domain['ANNUITY_INCOME_PERCENT'] = data_train_domain['AMT_ANNUITY'] / data_train_domain['AMT_INCOME_TOTAL']
    data_train_domain['CREDIT_TERM'] = data_train_domain['AMT_ANNUITY'] / data_train_domain['AMT_CREDIT']
    data_train_domain['DAYS_EMPLOYED_PERCENT'] = data_train_domain['DAYS_EMPLOYED'] / data_train_domain['DAYS_BIRTH']

    data_test_domain['CREDIT_INCOME_PERCENT'] = data_test_domain['AMT_CREDIT'] / data_test_domain['AMT_INCOME_TOTAL']
    data_test_domain['ANNUITY_INCOME_PERCENT'] = data_test_domain['AMT_ANNUITY'] / data_test_domain['AMT_INCOME_TOTAL']
    data_test_domain['CREDIT_TERM'] = data_test_domain['AMT_ANNUITY'] / data_test_domain['AMT_CREDIT']
    data_test_domain['DAYS_EMPLOYED_PERCENT'] = data_test_domain['DAYS_EMPLOYED'] / data_test_domain['DAYS_BIRTH']

    return data_train_domain, data_test_domain

def preprocesseur(df_train, df_test):
    
    # Cette fonction permet d'imputer les valeurs manquantes dans
    # chaque dataset et aussi d'appliquer un MinMaxScaler

    # Drop the target from the training data
    if "TARGET" in df_train:
        train = df_train.drop(columns = ["TARGET"])
    else:
        train = df_train.copy()
        
    # Feature names
    features = list(train.columns)


    # Median imputation of missing values
    imputer = SimpleImputer(strategy = 'median')

    # Scale each feature to 0-1
    scaler = MinMaxScaler(feature_range = (0, 1))

    # Replace the boolean column by numerics values 
    train["DAYS_EMPLOYED_ANOM"] = train["DAYS_EMPLOYED_ANOM"].astype("int")

    # Fit on the training data
    imputer.fit(train)

    # Transform both training and testing data
    train = imputer.transform(train)
    test = imputer.transform(df_test)

    # Repeat with the scaler
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    
    return train, test

def data_resampler(df_train, target):

    rsp = RandomUnderSampler()
    X_rsp, y_rsp = rsp.fit_resample(df_train, target["TARGET"])

    return X_rsp, y_rsp


# answer when asking for score and decision about one customer
@app.route('/scoring_cust/')
def scoring_cust():
    # Parse http request to get arguments (sk_id_cust)
    sk_id_cust = int(request.args.get('SK_ID_CURR'))
    # Get the data for the customer (pd.DataFrame)
    X_cust = X_test.loc[sk_id_cust:sk_id_cust]
	# Compute the score of the customer (using the whole pipeline)   
    score_cust = best_model.predict_proba(X_cust)[:,1][0]
    # Compute decision according to the best threshold (True: loan refused)
    bool_cust = (score_cust >= thresh)
    # Return processed data
    return jsonify({'status': 'ok',
    		        'SK_ID_CURR': sk_id_cust,
    		        'score': score_cust,
    		        'bool': str(bool_cust)})



def entrainement_lgbm(X, y):

    # Configuration de la meilleure itération trouvée par le RandomizeSearchCV
    # Optimized n_estimator=1144
    clf_lgbm = LGBM_final

    clf_lgbm.fit(X, y)

    return clf_lgbm

def entrainement_knn(df):

    print("En cours...")
    knn = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(df)

    return knn 

if __name__ == "__main__":
    app.run(host="localhost", port="5000", debug=True)

from joblib import load
from flask import Flask, jsonify, request, jsonify, render_template
import json
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling  import RandomUnderSampler
import xgboost as xgb
from lightgbm import LGBMClassifier as lgbm



# Create an instance of the Flask class that is the WSGI application.
# The first argument is the name of the application module or package,
# typically __name__ when using a single module.
app = Flask(__name__)


# On charge les données
data_train = pd.read_csv("Codes/data/application_train.csv")
data_test = pd.read_csv("Codes/data/application_test.csv")

# On charge le jeu d'entrainement et le modèle optimal
X_train = pickle.load(open('Codes/X_train_mediane.sav', 'rb'))
X_test = pickle.load(open('Codes/X_test_mediane.sav', 'rb'))

url_df = 'Codes/data_mediane.csv'
df_med = pd.read_csv(url_df)
df_med['SK_ID_CURR'] = df_med['SK_ID_CURR'].astype('int')

liste_clients = list(df_med['SK_ID_CURR'].sort_values())


LGBM_final = pickle.load(open('Codes/modele_lgbm_final.sav', 'rb'))

train = None
test = None
model = None

# On crée la liste des ID clients qui nous servira dans l'API
id_client = df_med["SK_ID_CURR"][:50].values
id_client = pd.DataFrame(id_client)

# routes
# Entraînement du modèle
@app.route("/init_model", methods=["GET"])
def init_model():
    
    # On prépare les données
    df_train, df_test = features_engineering(data_train, data_test)

    print("Features engineering done")
    
    # On fait le préprocessing des données
    df_train, df_test = preprocesseur(df_train, df_test)

    # On transforme le dataset de test préparé en variabe
    # globale, car il est utilisé dans la fonction predict
    global train
    train = df_train.copy()

    global test
    test = df_test.copy()

    print("Preprocessing done")
    
    # On fait un resampling des données d'entraînement
    X, y = data_resampler(df_train, test)
    print("Resampling done")

    # On entraîne le modèle et on le transforme en
    # variable globale pour la fonction predict
    global clf_lgbm
    clf_lgbm = entrainement_lgbm(X, y)
    print("Training lgbm done")

    global knn
    knn = entrainement_knn(df_train)
    print("Training knn done")

    return jsonify(["Initialisation terminee."])


@app.route('/<int:ID_client>', methods=["POST", "GET"])
def predict_client(ID_client):
    seuil = 0.66

    if ID_client in liste_clients:
        X = df_med[df_med['SK_ID_CURR'] == ID_client]
        X.drop('SK_ID_CURR', axis=1, inplace=True)

        predictions_proba = LGBM_final.predict_proba(X)
        predict_proba_0 = round(predictions_proba[0, 0], 3)
        predict_proba_1 = round(predictions_proba[0, 1], 3)

        if (1-predictions_proba[:, 1]) >= seuil:
            predict = 'Oui'
        elif (1-predictions_proba[:, 1]) < seuil:
            predict = 'Non'

         
    else:
        predict = 'Client inconnu'
        predict_proba_0 = 'Client inconnu'
        predict_proba_1 = 'Client inconnu'

    return jsonify({ 'retour_prediction' : str(predict),
                     'predict_proba_0': str(1-predict_proba_0),
                     'predict_proba_1': str(1-predict_proba_1) })
##########################################################################################

@app.route('/accord/<int:ID_client>', methods=["POST", "GET"])
def predict_client_accord(ID_client):
    seuil = 0.66

    if ID_client in liste_clients:
        X = df_med[df_med['SK_ID_CURR'] == ID_client]
        X.drop('SK_ID_CURR', axis=1, inplace=True)

        predictions_proba = LGBM_final.predict_proba(X)
        predict_proba_0 = round(predictions_proba[0, 0], 3)
        predict_proba_1 = round(predictions_proba[0, 1], 3)

        if (1-predictions_proba[:, 1]) >= seuil:
            predict = 'Oui'
        elif (1-predictions_proba[:, 1]) < seuil:
            predict = 'Non'

         
    else:
        predict = 'Client inconnu'
        predict_proba_0 = 'Client inconnu'
        predict_proba_1 = 'Client inconnu'

    return jsonify(str(predict) )

############################################################
@app.route('/score/<int:ID_client>', methods=["POST", "GET"])
def predict_client_score(ID_client):
    seuil = 0.66

    if ID_client in liste_clients:
        X = df_med[df_med['SK_ID_CURR'] == ID_client]
        X.drop('SK_ID_CURR', axis=1, inplace=True)

        predictions_proba = LGBM_final.predict_proba(X)
        predict_proba_0 = round(predictions_proba[0, 0], 3)
        predict_proba_1 = round(predictions_proba[0, 1], 3)
         
    else:
        predict = 'Client inconnu'
        predict_proba_0 = 'Client inconnu'
        predict_proba_1 = 'Client inconnu'

    return jsonify(str(1-predictions_proba[:,1][0]) )


# Chargement des données pour la selection de l'ID client
@app.route("/load_data", methods=["GET"])
def load_data():
    
    return id_client.to_json(orient='values')

# Chargement d'informations générales
@app.route("/infos_gen", methods=["GET"])
def infos_gen():

    lst_infos = [data_train.shape[0],
                 round(data_train["AMT_INCOME_TOTAL"].mean(), 2),
                 round(data_train["AMT_CREDIT"].mean(), 2)]

    return jsonify(lst_infos)

# Chargement des données pour le graphique
# dans la sidebar
@app.route("/disparite_target", methods=["GET"])
def disparite_target():

    df_target = data_train["TARGET"].value_counts()

    return df_target.to_json(orient='values')

# Chargement d'informations générales sur le client
@app.route("/infos_client", methods=["GET"])
def infos_client():

    id = request.args.get("id_client")

    data_client = data_test[data_test["SK_ID_CURR"] == id]

    print(data_client)
    
    dict_infos = {
#        "status_famille" : data_client["NAME_FAMILY_STATUS"].tolist(),
#        "nb_enfant" : data_client["CNT_CHILDREN"].tolist(),
# #        "age" : int(data_client["DAYS_BIRTH"].values / -365),
#        "revenus" : data_client["AMT_INCOME_TOTAL"].tolist(),
#        "montant_credit" : data_client["AMT_CREDIT"].tolist(),
#        "annuites" : data_client["AMT_ANNUITY"].tolist(),
#        "montant_bien" : data_client["AMT_GOODS_PRICE"].tolist()
       }
    
    response = json.loads(data_client.to_json(orient='index'))

    return response

# Calcul des ages de la population pour le graphique
# situant l'age du client
@app.route("/load_age_population", methods=["GET"])
def load_age_population():
    
    df_age = round((data_train["DAYS_BIRTH"] / -365), 2)
    return df_age.to_json(orient='values')

# Segmentation des revenus de la population pour le graphique
# situant l'age du client
@app.route("/load_revenus_population", methods=["GET"])
def load_revenus_population():
    
    # On supprime les outliers qui faussent le graphique de sortie
    # Cette opération supprime un peu moins de 300 lignes sur une
    # population > 300000...
    df_revenus = data_train[data_train["AMT_INCOME_TOTAL"] < 700000]
    
    df_revenus["tranches_revenus"] = pd.cut(df_revenus["AMT_INCOME_TOTAL"], bins=20)
    df_revenus = df_revenus[["AMT_INCOME_TOTAL", "tranches_revenus"]]
    df_revenus.sort_values(by="AMT_INCOME_TOTAL", inplace=True)

    print(df_revenus)
    
    df_revenus = df_revenus["AMT_INCOME_TOTAL"]

    return df_revenus.to_json(orient='values')

@app.route("/predict", methods=["GET"])
def predict():
    
    id = request.args.get("id_client")

    print("Analyse data_test :")
    print(data_test.shape)
    print(data_test[data_test["SK_ID_CURR"] == id]) #int(id)
     
    index = data_test[data_test["SK_ID_CURR"] == id].index.values

    print(index[0])
   
    data_client = test[index]

    print(data_client)

    prediction = clf_lgbm.predict_proba(data_client)

    prediction = prediction[0].tolist()

    print(prediction)

    return jsonify(prediction)

@app.route("/load_voisins", methods=["GET"])
def load_voisins():
    
    id = request.args.get("id_client")

    index = data_test[data_test["SK_ID_CURR"] == int(id)].index.values

    data_client = test[index]
    
    distances, indices = knn.kneighbors(data_client)

    print("indices")
    print(indices)
    print("distances")
    print(distances)

    df_voisins = data_train.iloc[indices[0], :]
    
    response = json.loads(df_voisins.to_json(orient='index'))

    return response


def features_engineering(data_train, data_test):

    # Cette fonction regroupe toutes les opérations de features engineering
    # mises en place sur les sets train & test

    #############################################
    # LABEL ENCODING
    #############################################
    # Create a label encoder object
    le = LabelEncoder()
    le_count = 0

    # Iterate through the columns
    for col in data_train.columns:
        if data_train[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(data_train[col].unique())) <= 2:
                # Train on the training data
                le.fit(data_train[col])
                # Transform both training and testing data
                data_train[col] = le.transform(data_train[col])
                data_test[col] = le.transform(data_test[col])
                
                # Keep track of how many columns were label encoded
                le_count += 1

    ############################################
    # ONE HOT ENCODING
    ############################################
    # one-hot encoding of categorical variables
    data_train = pd.get_dummies(data_train)
    data_test = pd.get_dummies(data_test)

    train_labels = data_train['TARGET']
    # Align the training and testing data, keep only columns present in both dataframes
    data_train, data_test = data_train.align(data_test, join = 'inner', axis = 1)
    # Add the target back in
    data_train['TARGET'] = train_labels

    ############################################
    # VALEURS ABERRANTES
    ############################################
    # Create an anomalous flag column
    data_train['DAYS_EMPLOYED_ANOM'] = data_train["DAYS_EMPLOYED"] == 365243
    # Replace the anomalous values with nan
    data_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
    data_test['DAYS_EMPLOYED_ANOM'] = data_test["DAYS_EMPLOYED"] == 365243
    data_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

    # Traitement des valeurs négatives
    data_train['DAYS_BIRTH'] = abs(data_train['DAYS_BIRTH'])

    ############################################
    # CREATION DE VARIABLES

    # In this frame of mind, we can make a couple features that attempt to capture what we think 
    # may be important for telling whether a client will default on a loan. 
    # Here I'm going to use five features that were inspired by this script by Aguiar:

    # CREDIT_INCOME_PERCENT: the percentage of the credit amount relative to a client's income
    # ANNUITY_INCOME_PERCENT: the percentage of the loan annuity relative to a client's income
    # CREDIT_TERM: the length of the payment in months (since the annuity is the monthly amount due
    # DAYS_EMPLOYED_PERCENT: the percentage of the days employed relative to the client's age
    # Again, thanks to Aguiar and his great script for exploring these features.
    
    data_train_domain = data_train.copy()
    data_test_domain = data_test.copy()

    data_train_domain['CREDIT_INCOME_PERCENT'] = data_train_domain['AMT_CREDIT'] / data_train_domain['AMT_INCOME_TOTAL']
    data_train_domain['ANNUITY_INCOME_PERCENT'] = data_train_domain['AMT_ANNUITY'] / data_train_domain['AMT_INCOME_TOTAL']
    data_train_domain['CREDIT_TERM'] = data_train_domain['AMT_ANNUITY'] / data_train_domain['AMT_CREDIT']
    data_train_domain['DAYS_EMPLOYED_PERCENT'] = data_train_domain['DAYS_EMPLOYED'] / data_train_domain['DAYS_BIRTH']

    data_test_domain['CREDIT_INCOME_PERCENT'] = data_test_domain['AMT_CREDIT'] / data_test_domain['AMT_INCOME_TOTAL']
    data_test_domain['ANNUITY_INCOME_PERCENT'] = data_test_domain['AMT_ANNUITY'] / data_test_domain['AMT_INCOME_TOTAL']
    data_test_domain['CREDIT_TERM'] = data_test_domain['AMT_ANNUITY'] / data_test_domain['AMT_CREDIT']
    data_test_domain['DAYS_EMPLOYED_PERCENT'] = data_test_domain['DAYS_EMPLOYED'] / data_test_domain['DAYS_BIRTH']

    return data_train_domain, data_test_domain

def preprocesseur(df_train, df_test):
    
    # Cette fonction permet d'imputer les valeurs manquantes dans
    # chaque dataset et aussi d'appliquer un MinMaxScaler

    # Drop the target from the training data
    if "TARGET" in df_train:
        train = df_train.drop(columns = ["TARGET"])
    else:
        train = df_train.copy()
        
    # Feature names
    features = list(train.columns)


    # Median imputation of missing values
    imputer = SimpleImputer(strategy = 'median')

    # Scale each feature to 0-1
    scaler = MinMaxScaler(feature_range = (0, 1))

    # Replace the boolean column by numerics values 
    train["DAYS_EMPLOYED_ANOM"] = train["DAYS_EMPLOYED_ANOM"].astype("int")

    # Fit on the training data
    imputer.fit(train)

    # Transform both training and testing data
    train = imputer.transform(train)
    test = imputer.transform(df_test)

    # Repeat with the scaler
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)
    
    return train, test

def data_resampler(df_train, target):

    rsp = RandomUnderSampler()
    X_rsp, y_rsp = rsp.fit_resample(df_train, target["TARGET"])

    return X_rsp, y_rsp


# answer when asking for score and decision about one customer
@app.route('/scoring_cust/')
def scoring_cust():
    # Parse http request to get arguments (sk_id_cust)
    sk_id_cust = int(request.args.get('SK_ID_CURR'))
    # Get the data for the customer (pd.DataFrame)
    X_cust = X_test.loc[sk_id_cust:sk_id_cust]
	# Compute the score of the customer (using the whole pipeline)   
    score_cust = best_model.predict_proba(X_cust)[:,1][0]
    # Compute decision according to the best threshold (True: loan refused)
    bool_cust = (score_cust >= thresh)
    # Return processed data
    return jsonify({'status': 'ok',
    		        'SK_ID_CURR': sk_id_cust,
    		        'score': score_cust,
    		        'bool': str(bool_cust)})



def entrainement_lgbm(X, y):

    # Configuration de la meilleure itération trouvée par le RandomizeSearchCV
    # Optimized n_estimator=1144
    clf_lgbm = LGBM_final

    clf_lgbm.fit(X, y)

    return clf_lgbm

def entrainement_knn(df):

    print("En cours...")
    knn = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(df)

    return knn 

if __name__ == "__main__":
    app.run(host="localhost", port="5000", debug=True)
