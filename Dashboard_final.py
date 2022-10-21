import streamlit as st

import requests
import json

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from flask import Flask
import plotly.graph_objects as go
from streamlit_shap import st_shap
import shap
import plotly.express as px
import numpy as np
import pickle
import altair as alt


URL_API = "http://localhost:5000/"

data_train = pd.read_csv("Codes/data/application_train.csv")
data_test = pd.read_csv("Codes/data/application_test.csv")


url_df = 'Codes/data_mediane.csv'
df_med = pd.read_csv(url_df)
df_med['SK_ID_CURR'] = df_med['SK_ID_CURR'].astype('int')

liste_clients = list(df_med['SK_ID_CURR'].sort_values())
new_df=df_med
best_model = pickle.load(open('Codes/modele_lgbm_final.sav', 'rb'))

def main():
   
    init = st.markdown("*Initialisation de l'application en cours...*")
#    init = st.markdown(init_api())

   
    # Display the title
    st.title('Décision d\'octroi de crédit')
    st.markdown("Interprétation des prédictions faites par le modèle afin de justifier la décision d’octroi du crédit au client.")
    img = Image.open("Credit.png")
    st.image(img, width=200)
    # Affichage d'informations dans la sidebar
    st.sidebar.subheader("Informations générales")
    # Chargement du logo
    logo = load_logo()
    st.sidebar.image(logo,
                     width=200)

    # Chargement de la selectbox
    lst_id = load_selectbox()
    global id_client
    id_client = st.sidebar.selectbox("ID Client", liste_clients)
    
    # Chargement des infos générales
    nb_credits, rev_moy, credits_moy, targets = load_infos_gen()

    # Affichage des infos générales dans la sidebar
    # Nombre de crédits existants
    st.sidebar.markdown("<u>Nombre crédits existants dans la base :</u>", unsafe_allow_html=True)
    st.sidebar.text(nb_credits)

    # Graphique camembert
    st.sidebar.markdown("<u>Différence solvabilité / non solvabilité</u>", unsafe_allow_html=True)

    plt.pie(targets, explode=[0, 0.1], labels=["Solvable", "Non solvable"], autopct='%1.1f%%',
            shadow=True, startangle=90)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.sidebar.pyplot()

    # Revenus moyens
    st.sidebar.markdown("<u>Revenus moyens $(USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(rev_moy)

    # Montant crédits moyen
    st.sidebar.markdown("<u>Montant crédits moyen $(USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(credits_moy)
    
    # Affichage de l'ID client sélectionné
    st.write("Vous avez sélectionné le client :", id_client)
    
        # Affichage état civil
    st.header("**Informations client**")
    #infos = st.checkbox("Afficher les informations du client?")
    
    data= data_train.loc[data_train['SK_ID_CURR'] == id_client]
     
    nombre_enfants = data["CNT_CHILDREN"].values
    statut_familial=data['NAME_FAMILY_STATUS'].values
    genre = data['CODE_GENDER'].values
    age= data["DAYS_BIRTH"].values / -365
    revenus= data["AMT_INCOME_TOTAL"].values
    annuites = data["AMT_ANNUITY"].values
    montant_bien = data["AMT_GOODS_PRICE"].values
    voiture=data['FLAG_OWN_CAR'].values    


    if st.checkbox("Afficher les informations du client "):
     
        st.write("Le client est ", "une Femme" if genre=='F' else "un Homme" )
        st.write("L'age du client est de  :", int(age) , "ans.")
        st.write("Statut familial : ",str(statut_familial))
        st.write("Nombre d'enfant(s) : ", int(nombre_enfants))
        st.write("Revenus : ", float(revenus))
        st.write("Montant du bien : ",float(montant_bien))
        st.write("Annuites : ", float(annuites))
        st.write("Le client possède une voiture : ",'Oui' if voiture == 1 else 'Non')
        
    else:
        st.markdown("<i>Informations masquées</i>", unsafe_allow_html=True)
        
       
    # Affichage octroie du credit                
    st.header("**Octroi du crédit**")
    
 #################### ACCORD CREDIT ##########################

    lien_accord = URL_API  + f"/accord/{id_client}"
    # Requesting the API and saving the response
    response = requests.get(lien_accord)
    # Convert from JSON format to Python dict
    content = json.loads(response.content.decode('utf-8'))

    if content=="Oui":
        st.success("Crédit accordé ")
    elif content=="Non": 
        st.error("Crédit non accordé")
    else: 
        st.warning("Erreur ")
    
    
    lien_score = URL_API  + f"/score/{id_client}"
    # Requesting the API and saving the response
    response1 = requests.get(lien_score)
    # Convert from JSON format to Python dict
    score_model = json.loads(response1.content.decode('utf-8'))

    seuil = 0.66
    seuil=float(seuil)
    score_model=float(score_model)
    
    
    st.header('Score de solvabilité')
    if score_model < seuil:
         fig = go.Figure(go.Indicator(
                             mode = 'gauge + number',
                             value = score_model,
                             domain = {'x': [0, 1], 'y': [0, 1]},
                             delta = {'reference': seuil},
                             gauge = {'axis': {'range': [0, 1]},
                                        'bar': {'color': 'red'},
                                     'steps' : [{'range': [0, seuil], 'color': "lightgrey"},
                                                {'range': [seuil, 1], 'color': "grey"}],
                                     'threshold' : {'line': {'color': 'green', 'width': 4}, 'thickness': 0.75, 'value': seuil}}
                         ))
    else:
         fig = go.Figure(go.Indicator(
                             mode = 'gauge + number',
                             value = score_model,
                             domain = {'x': [0, 1], 'y': [0, 1]},
                             delta = {'reference': seuil},
                             gauge = {'axis': {'range': [0, 1]},
                                     'bar': {'color': 'green'},
                                     'steps' : [{'range': [0, seuil], 'color': "grey"},
                                                 {'range': [seuil, 1], 'color': "lightgrey"}],
                                     'threshold' : {'line': {'color': 'black', 'width': 4}, 'thickness': 0.75, 'value': seuil}}
                         ))
    st.plotly_chart(fig)

######################## SCORE MODEL ########################

   
    tab1, tab2, tab3 = st.tabs(['Impacte des variables sur le score client', 'Impacte des variable (ensemble de clients)', 'Comparaison aux autres clients'])
    with tab1:
        # Shap values
        explainer = shap.TreeExplainer(best_model)
        df_api_url = "Codes/data_mediane.csv"
        df_API = pd.read_csv(df_api_url)
        df_shap = df_API.loc[:, df_API.columns != 'SK_ID_CURR']
        shap_values = explainer.shap_values(df_shap)
    
        # Interprétation pour l'individu choisi
        st.subheader("Impact des variables sur le score pour le client " + str(id_client))
        id = df_med[df_med['SK_ID_CURR'] == id_client].index
        st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][id, :], df_shap.iloc[id, :], link='logit'))
        st.write('Les variables en bleu ont contribué à accorder le crédit (donc à augmenter le score).\n Les variables en rose ont contribué à refuser le crédit (donc à diminuer le score)')
   
    with tab2:
        # Interprétation pour l'ensemble des clients
        st.header("Impact des variables pour l'ensemble des clients")
        st.write('Les variables en bleu ont contribué à accorder le crédit (donc à augmenter le score).\n Les variables en rose ont contribué à refuser le crédit (donc à diminuer le score)')
        st_shap(shap.summary_plot(shap_values, df_shap))

    
    
    
    with tab3:
        st.header("Comparaison aux autres clients")
        categ = ['CODE_GENDER', 'FLAG_OWN_CAR', 'NAME_EDUCATION_TYPE_Highereducation']
       
        col1, col2 = st.columns(2)
        with col1:
            liste_variables1 = ['Revenus par personne', 'ID_Client', 'Genre', 'Âge', 'Ancienneté de l\'emploi', 'Revenus totaux',
                                'Nombre de personnes dans la famille', 'Voiture personnelle', 'Education secondaire',
                                'Montant des produits à l\'origine de la demande de prêt', 'Montant du crédit',
                                'Fréquence de paiement', 'Montant des annuités', 'Source externe 2', 'Source externe 3']
                   
            variable1 = st.selectbox("Sélectionnez la première variable à afficher", liste_variables1, key=1)
            var_en1 = get_english_var(variable1)
            if var_en1 in categ:
                var1_cat = 1
            else:
                var1_cat = 0
       
        with col2:
            liste_variables2 = ['Ancienneté de l\'emploi', 'ID_Client', 'Genre', 'Âge', 'Revenus totaux',
                                'Nombre de personnes dans la famille', 'Revenus par personne', 'Voiture personnelle',
                                'Education secondaire', 'Montant des produits à l\'origine de la demande de prêt', 'Montant du crédit',
                                'Fréquence de paiement', 'Montant des annuités', 'Source externe 2', 'Source externe 3']
            
            variable2 = st.selectbox("Sélectionnez la seconde variable à afficher", liste_variables2, key=2)
            var_en2 = get_english_var(variable2)
            if var_en2 in categ:
                var2_cat = 1
            else:
                var2_cat = 0
       
        df_comp = pd.read_csv('Codes/data_mediane.csv')

        if variable1 == variable2:
            df_comp = df_comp[[var_en1, 'TARGET']].dropna()
        else:  
            df_comp = df_comp[[var_en1, var_en2, 'TARGET']].dropna()
       
        col1_, col2_ = st.columns(2)
        with col1_:
            if var1_cat == 0:
                marg = 'box'
            else:
                marg = None
            fig1 = px.histogram(df_comp, x=var_en1, color='TARGET', marginal=marg, nbins=50)
            if var1_cat == 0:
                fig1.add_vline(x=new_df[var_en1].mean(), line_width=5, line_color='#8f00ff', name='Client ' + str(id_client))
            fig1.update_layout(barmode='overlay', title={'text': variable1, 'x': 0.5, 'xanchor': 'center'})
            fig1.update_traces(opacity=0.75)
            st.plotly_chart(fig1, use_container_width=True)
        with col2_:
            if var2_cat == 0:
                marg = 'box'
            else:
                marg = None
            fig2 = px.histogram(df_comp, x=var_en2, color='TARGET', marginal=marg, nbins=50)
            if var2_cat == 0:
                fig2.add_vline(x=new_df[var_en2].mean(), line_width=5, line_color='#8f00ff', name='Client ' + str(id_client))
            fig2.update_layout(barmode='overlay', title={'text': variable2, 'x': 0.5, 'xanchor': 'center'})
            fig2.update_traces(opacity=0.75)
            st.plotly_chart(fig2, use_container_width=True)


        st.line_chart(data=new_df, x=var_en1,y=var_en2)
       

           
       
        
            
            
def fetch(session, url):
    try:
        result = session.get(url)
        return result.json()
    except Exception:
        return {}    
########################### features importances    
    # Functions
    # ----------
    def get_list_display_features(f, def_n, key):
        all_feat = f
        n = st.slider("Nb of features to display",
                      min_value=2, max_value=40,
                      value=def_n, step=None, format=None, key=key)

        disp_cols = list(get_features_importances().sort_values(ascending=False).iloc[:n].index)

        box_cols = st.multiselect(
            'Choose the features to display:',
            sorted(all_feat),
            default=disp_cols, key=key)
        return box_cols
#######################
  #              Display feature's distribution (Boxplots)
        ##########################################################################
        if st.checkbox('show features distribution by class', key=20):
            st.header('Boxplots of the main features')
            fig, ax = plt.subplots(figsize=(20, 10))
            with st.spinner('Boxplot creation in progress...please wait.....'):
                # Get Shap values for customer
                shap_vals, expected_vals = values_shap(selected_id)
                # Get features names
                features = feat()
                # Get selected columns
                disp_box_cols = get_list_display_features(features, 2, key=45)
                # -----------------------------------------------------------------------------------------------
                # Get tagets and data for : all customers + Applicant customer + 20 neighbors of selected customer
                # -----------------------------------------------------------------------------------------------
                # neighbors + Applicant customer :
                data_neigh, target_neigh = get_data_neigh(selected_id)
                data_thousand_neigh, target_thousand_neigh, x_customer = get_data_thousand_neigh(selected_id)

                x_cust, y_cust = get_selected_cust_data(selected_id)
                x_customer.columns = x_customer.columns.str.split('.').str[0]
                # Target impuatation (0 : 'repaid (....), 1 : not repaid (....)
                # -------------------------------------------------------------
                target_neigh = target_neigh.replace({0: 'repaid (neighbors)',
                                                     1: 'not repaid (neighbors)'})
                target_thousand_neigh = target_thousand_neigh.replace({0: 'repaid (neighbors)',
                                                                       1: 'not repaid (neighbors)'})
                y_cust = y_cust.replace({0: 'repaid (customer)',
                                         1: 'not repaid (customer)'})

                # y_cust.rename(columns={'10006':'TARGET'}, inplace=True)
                # ------------------------------
                # Get 1000 neighbors personal data
                # ------------------------------
                df_thousand_neigh = pd.concat([data_thousand_neigh[disp_box_cols], target_thousand_neigh], axis=1)
                df_melt_thousand_neigh = df_thousand_neigh.reset_index()
                df_melt_thousand_neigh.columns = ['index'] + list(df_melt_thousand_neigh.columns)[1:]
                df_melt_thousand_neigh = df_melt_thousand_neigh.melt(id_vars=['index', 'TARGET'],
                                                                     value_vars=disp_box_cols,
                                                                     var_name="variables",  # "variables",
                                                                     value_name="values")

                sns.boxplot(data=df_melt_thousand_neigh, x='variables', y='values',
                            hue='TARGET', linewidth=1, width=0.4,
                            palette=['tab:green', 'tab:red'], showfliers=False,
                            saturation=0.5, ax=ax)
           
        
        
  #                         Model's decision checkbox
    ##############################################################################
    
def get_english_var(var_fr):
    liste_var_en = ['SK_ID_CURR', 'CODE_GENDER', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL', 'CNT_FAM_MEMBERS',
                    'INCOME_PER_PERSON', 'FLAG_OWN_CAR', 'NAME_EDUCATION_TYPE_Highereducation', 'AMT_GOODS_PRICE', 'AMT_CREDIT',
                    'PAYMENT_RATE', 'AMT_ANNUITY', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    liste_var_fr = ['ID_Client', 'Genre', 'Âge', 'Ancienneté de l\'emploi', 'Revenus totaux', 'Nombre de personnes dans la famille',
                    'Revenus par personne', 'Voiture personnelle', 'Education secondaire',
                    'Montant des produits à l\'origine de la demande de prêt', 'Montant du crédit', 'Fréquence de paiement',
                    'Montant des annuités', 'Source externe 2', 'Source externe 3']
    ind = liste_var_fr.index(var_fr)
    var_en = liste_var_en[ind]
    return var_en

@st.cache
def get_score_model():
    # URL of the sk_id API
    score_api_url = URL_API + "scoring_cust/?SK_ID_CURR=" + str('SK_ID_CURR')
    # Requesting the API and saving the response
    response = requests.get(score_api_url)
    # Convert from JSON format to Python dict
    content = json.loads(response.content.decode('utf-8'))
    # Getting the values of "ID" from the content
    score_model = (content['score'])
    threshold = content['thresh']
    return score_model, threshold

  
@st.cache
def init_api():

    # Requête permettant de récupérer la liste des ID clients
    init_api = requests.get(URL_API + "init_model")
    init_api = init_api.json()

    return "Initialisation application terminée."

    
    
    #########################################
@st.cache
def init_api():

    # Requête permettant de récupérer la liste des ID clients
    init_api = requests.get(URL_API + "init_model")
    init_api = init_api.json()

    return "Initialisation application terminée."

@st.cache()
def load_logo():
    # Construction de la sidebar
    # Chargement du logo
    logo = Image.open("logo.png") 
    
    return logo

@st.cache()
def load_selectbox():
    # Requête permettant de récupérer la liste des ID clients
    data_json = requests.get(URL_API + "load_data")
    data = data_json.json()

    # Récupération des valeurs sans les [] de la réponse
    lst_id = []
    for i in data:
        lst_id.append(i[0])

    return lst_id

@st.cache()
def load_infos_gen():

    # Requête permettant de récupérer :
    # Le nombre de lignes de crédits existants dans la base
    # Le revenus moyens des clients
    # Le montant moyen des crédits existants
    infos_gen = requests.get(URL_API + "infos_gen")
    infos_gen = infos_gen.json()

    nb_credits = infos_gen[0]
    rev_moy = infos_gen[1]
    credits_moy = infos_gen[2]

    # Requête permettant de récupérer
    # Le nombre de target dans la classe 0
    # et la classe 1
    targets = requests.get(URL_API + "disparite_target")    
    targets = targets.json()


    return nb_credits, rev_moy, credits_moy, targets

@st.cache
def load_age_population():
    
    # Requête permettant de récupérer les âges de la 
    # population pour le graphique situant le client
    data_age_json = requests.get(URL_API + "load_age_population")
    data_age = data_age_json.json()

    return data_age

@st.cache
def load_revenus_population():
    
    # Requête permettant de récupérer des tranches de revenus 
    # de la population pour le graphique situant le client
    data_revenus_json = requests.get(URL_API + "load_revenus_population")
    
    data_revenus = data_revenus_json.json()

    return data_revenus

def load_prediction():
    
    # Requête permettant de récupérer la prédiction
    # de faillite du client sélectionné
    prediction = requests.get(URL_API + "predict", params={"id_client":id_client})
#     prediction = json.loads(prediction.content.decode('utf-8'))

#     ///////////////////////////////
#     prediction = json.loads(prediction)
#     file = open(local_path,'rb')

#     prediction.raise_for_status()
#     with open(prediction, 'rb') as p:
#         prediction = json.loads(p.read())
#          headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:43.0) Gecko/20100101 Firefox/43.0'}

# #     ////////////////////////////////////////
   
#     prediction = pd.DataFrame(prediction) 
    prediction = prediction.json()
    return prediction[1]
    


def load_voisins():
    
    # Requête permettant de récupérer les 10 dossiers
    # les plus proches de l'ID client choisi
    voisins = requests.get(URL_API + "load_voisins", params={"id_client":id_client})

    # On transforme la réponse en dictionnaire python
    voisins = json.loads(voisins.content.decode("utf-8"))
    
    # On transforme le dictionnaire en dataframe
    voisins = pd.DataFrame.from_dict(voisins).T

    # On déplace la colonne TARGET en premier pour plus de lisibilité
    target = voisins["TARGET"]
    voisins.drop(labels=["TARGET"], axis=1, inplace=True)
    voisins.insert(0, "TARGET", target)
    
    return voisins

if __name__ == "__main__":
    main()
