import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import psycopg2
from sqlalchemy import create_engine
import boto3
from io import StringIO
import os

# Configuration de la page
st.set_page_config(
    page_title="InvestScan - Analyse Immobilière",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre et introduction
st.title("🏠 InvestScan - Plateforme d'analyse immobilière")
st.markdown("""
Cette application permet d'analyser les données immobilières pour identifier les meilleures opportunités d'investissement.
""")


# Fonctions utilitaires
def connect_to_postgres():
    """Établit une connexion à la base de données PostgreSQL."""
    try:
        conn = psycopg2.connect(
            host="postgres",
            database="airflow",
            user="airflow",
            password="airflow"
        )
        return conn
    except Exception as e:
        st.error(f"Erreur de connexion à PostgreSQL: {e}")
        return None


def connect_to_minio():
    """Établit une connexion à MinIO (compatible S3)."""
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url='http://minio:9000',
            aws_access_key_id='minioadmin',
            aws_secret_access_key='minioadmin',
            region_name='us-east-1',
            config=boto3.session.Config(signature_version='s3v4')
        )
        return s3_client
    except Exception as e:
        st.error(f"Erreur de connexion à MinIO: {e}")
        return None


# Barre latérale avec les filtres
st.sidebar.header("Filtres")
selected_view = st.sidebar.selectbox(
    "Sélectionner une vue",
    ["Vue d'ensemble", "Analyse des prix", "Rentabilité", "Cartographie", "À propos"]
)

# Afficher des données d'exemple si les connexions ne sont pas disponibles
if selected_view == "Vue d'ensemble":
    st.header("Vue d'ensemble du marché immobilier")

    # Essayer de charger les données réelles
    conn = connect_to_postgres()
    if conn:
        try:
            # Requête pour obtenir des données réelles
            sql = """
            SELECT * FROM bien_immobilier LIMIT 100
            """
            df = pd.read_sql(sql, conn)
            conn.close()
            st.success("Connexion à la base de données établie avec succès!")
        except Exception as e:
            st.warning(f"Impossible de charger les données: {e}")
            # Générer des données d'exemple
            np.random.seed(42)
            df = pd.DataFrame({
                'id': range(1, 101),
                'type': np.random.choice(['Appartement', 'Maison', 'Villa', 'Studio'], 100),
                'prix': np.random.normal(250000, 100000, 100),
                'surface': np.random.normal(80, 30, 100),
                'nb_pieces': np.random.choice([1, 2, 3, 4, 5], 100),
                'code_postal': np.random.choice(['75001', '75002', '75003', '69001', '69002', '13001', '13002'], 100),
                'ville': np.random.choice(['Paris', 'Lyon', 'Marseille'], 100),
                'date_creation': pd.date_range(start='2023-01-01', periods=100)
            })
            st.info("Utilisation de données d'exemple")
    else:
        # Générer des données d'exemple
        np.random.seed(42)
        df = pd.DataFrame({
            'id': range(1, 101),
            'type': np.random.choice(['Appartement', 'Maison', 'Villa', 'Studio'], 100),
            'prix': np.random.normal(250000, 100000, 100),
            'surface': np.random.normal(80, 30, 100),
            'nb_pieces': np.random.choice([1, 2, 3, 4, 5], 100),
            'code_postal': np.random.choice(['75001', '75002', '75003', '69001', '69002', '13001', '13002'], 100),
            'ville': np.random.choice(['Paris', 'Lyon', 'Marseille'], 100),
            'date_creation': pd.date_range(start='2023-01-01', periods=100)
        })
        st.info("Utilisation de données d'exemple")

    # Calculer quelques métriques
    if 'prix' in df.columns and 'surface' in df.columns:
        df['prix_m2'] = df['prix'] / df['surface']

    # Afficher des KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nombre de biens", f"{len(df)}")
    with col2:
        if 'prix' in df.columns:
            st.metric("Prix moyen", f"{df['prix'].mean():,.0f} €")
    with col3:
        if 'surface' in df.columns:
            st.metric("Surface moyenne", f"{df['surface'].mean():,.0f} m²")
    with col4:
        if 'prix_m2' in df.columns:
            st.metric("Prix moyen au m²", f"{df['prix_m2'].mean():,.0f} €/m²")

    # Afficher un tableau des données
    st.subheader("Aperçu des données")
    st.dataframe(df)

    # Visualisations
    st.subheader("Répartition des prix")
    if 'prix' in df.columns and 'type' in df.columns:
        fig = px.histogram(df, x="prix", color="type", nbins=30,
                           title="Distribution des prix par type de bien")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Prix en fonction de la surface")
    if 'prix' in df.columns and 'surface' in df.columns and 'type' in df.columns:
        fig = px.scatter(df, x="surface", y="prix", color="type",
                         hover_data=["ville", "nb_pieces"],
                         title="Relation prix/surface par type de bien")
        st.plotly_chart(fig, use_container_width=True)

elif selected_view == "Analyse des prix":
    st.header("Analyse détaillée des prix")
    st.write("Cette section permettra d'analyser en détail les prix par zone géographique.")

    # Ici, vous pourriez ajouter des analyses plus approfondies
    st.info("Fonctionnalité en cours de développement")

elif selected_view == "Rentabilité":
    st.header("Analyse de rentabilité")
    st.write("Cette section permettra d'évaluer la rentabilité des investissements immobiliers.")

    # Ajoutez ici vos analyses de rentabilité
    st.info("Fonctionnalité en cours de développement")

elif selected_view == "Cartographie":
    st.header("Cartographie des biens immobiliers")
    st.write("Cette section affichera une carte interactive des biens immobiliers.")

    # Ici, vous pourriez intégrer une carte avec pydeck ou folium
    st.info("Fonctionnalité en cours de développement")

elif selected_view == "À propos":
    st.header("À propos de InvestScan")
    st.write("""
    InvestScan est une plateforme d'analyse immobilière développée dans le cadre de la certification 
    "Expert(e) en science des données". Elle utilise une architecture moderne basée sur:

    - Apache Airflow pour l'orchestration
    - Apache NiFi pour l'ETL
    - Apache Spark pour le traitement de données
    - PostgreSQL pour le stockage relationnel
    - MinIO pour le stockage d'objets
    - Kafka pour les flux temps réel
    - Streamlit pour la visualisation

    Cette architecture permet d'analyser efficacement les données immobilières pour identifier 
    les meilleures opportunités d'investissement.
    """)

# Pied de page
st.markdown("---")
st.caption("© 2025 InvestScan | Données à titre d'illustration uniquement")