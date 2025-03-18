import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
import json
import requests
import geopandas as gpd
from shapely.geometry import Point
from minio import Minio
from utils.data_loader import load_data_from_minio, load_geojson_from_minio
from utils.model_loader import load_sklearn_model, predict_price_sklearn
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Immobilière",
    page_icon="🏠",
    layout="wide"
)

# Fonction pour obtenir les coordonnées GPS à partir d'une adresse
def get_coordinates_from_address(address):
    """
    Utilise l'API de géocodage pour obtenir les coordonnées GPS d'une adresse
    """
    # Utiliser l'API de géocodage du gouvernement français (API Adresse)
    base_url = "https://api-adresse.data.gouv.fr/search/"
    params = {
        "q": address,
        "limit": 1
    }

    try:
        response = requests.get(base_url, params=params)
        data = response.json()

        if data["features"]:
            feature = data["features"][0]
            coordinates = feature["geometry"]["coordinates"]
            properties = feature["properties"]

            return {
                "lat": coordinates[1],
                "lon": coordinates[0],
                "city": properties.get("city", ""),
                "postcode": properties.get("postcode", ""),
                "street": properties.get("name", "")
            }
        else:
            return None
    except Exception as e:
        st.error(f"Erreur lors de la géolocalisation: {e}")
        return None


# Fonction pour charger les contours IRIS
@st.cache_data
def load_iris_contours():
    """
    Charge les contours IRIS depuis MinIO
    """
    try:
        # Charger les contours IRIS depuis MinIO
        gdf = load_geojson_from_minio(
            bucket_name="raw-data",
            object_name="iris-contours/20250227/iris_contours.gpkg"
        )
        return gdf
    except Exception as e:
        st.error(f"Erreur lors du chargement des contours IRIS: {e}")
        # Créer un GeoDataFrame vide en cas d'erreur
        return gpd.GeoDataFrame()


def get_iris_code_from_coordinates(lat, lon, iris_contours):
    """
    Détermine le code IRIS correspondant aux coordonnées GPS
    """
    try:
        # Vérifier si le GeoDataFrame est vide
        if iris_contours.empty:
            st.warning("Le fichier des contours IRIS est vide")
            return None

        # Créer un point géographique à partir des coordonnées
        point = Point(lon, lat)

        # S'assurer que le système de coordonnées est en WGS84
        if iris_contours.crs is None:
            iris_contours.set_crs(epsg=4326, inplace=True)
        elif iris_contours.crs.to_epsg() != 4326:
            iris_contours = iris_contours.to_crs(epsg=4326)

        # Corriger les géométries invalides si nécessaire
        if iris_contours.geometry.is_valid.sum() != len(iris_contours):
            iris_contours['geometry'] = iris_contours.geometry.buffer(0)

        # Trouver l'IRIS qui contient ce point - Méthode optimisée
        # C'est beaucoup plus efficace que d'itérer sur toutes les lignes
        point_gdf = gpd.GeoDataFrame([{"geometry": point}], geometry="geometry", crs="EPSG:4326")

        # Essayer une jointure spatiale standard
        try:
            spatial_join = gpd.sjoin(point_gdf, iris_contours, how="left", predicate="within")

            if not spatial_join.empty and not spatial_join.iloc[0].isna().all():
                # Récupérer les infos de l'IRIS correspondant
                result = {
                    "iris_code": spatial_join.iloc[0].get("code_iris", ""),
                    "iris_name": spatial_join.iloc[0].get("nom_iris", ""),
                    "commune": spatial_join.iloc[0].get("nom_commune", "")
                }
                return result
        except Exception as e:
            st.warning(f"Premier essai de jointure spatiale échoué: {e}")

        # Si ça ne fonctionne pas, essayer avec un léger buffer autour du point
        try:
            # Ajouter un petit buffer autour du point (~10m)
            point_buffered = Point(lon, lat).buffer(0.0001)
            point_gdf = gpd.GeoDataFrame([{"geometry": point_buffered}], geometry="geometry", crs="EPSG:4326")

            spatial_join = gpd.sjoin(point_gdf, iris_contours, how="left", predicate="intersects")

            if not spatial_join.empty and not spatial_join.iloc[0].isna().all():
                # Récupérer les infos de l'IRIS correspondant
                result = {
                    "iris_code": spatial_join.iloc[0].get("code_iris", ""),
                    "iris_name": spatial_join.iloc[0].get("nom_iris", ""),
                    "commune": spatial_join.iloc[0].get("nom_commune", "")
                }
                return result
        except Exception as e:
            st.warning(f"Deuxième essai de jointure spatiale échoué: {e}")

        # Si tout échoue, rechercher le plus proche manuellement
        min_distance = float('inf')
        closest_iris = None

        for idx, row in iris_contours.iterrows():
            if row.geometry.is_valid:
                distance = row.geometry.distance(point)
                if distance < min_distance:
                    min_distance = distance
                    closest_iris = {
                        "iris_code": row.get("code_iris", ""),
                        "iris_name": row.get("nom_iris", ""),
                        "commune": row.get("nom_commune", "")
                    }

        # Si un IRIS proche a été trouvé et qu'il est à moins de 0.005 degré (~500m)
        if closest_iris and min_distance < 0.005:
            st.info(
                f"Point hors des limites IRIS. Utilisation du plus proche à {min_distance:.5f} degrés (environ {min_distance * 111000:.0f}m).")
            return closest_iris

        # Aucun IRIS trouvé
        st.warning("Aucun IRIS trouvé pour ces coordonnées.")
        return None

    except Exception as e:
        st.error(f"Erreur lors de la recherche du code IRIS: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None


# Fonction pour charger les données socio-économiques par IRIS
@st.cache_data
def load_socioeconomic_data():
    """
    Charge les données socio-économiques par IRIS depuis MinIO
    """
    try:
        # Charger les données depuis MinIO
        df = load_data_from_minio(
            bucket_name="raw-data",
            object_name="base-td-file-iris/20250316/base_td_file_iris.csv",
            file_format="csv",
            sep=";"
        )
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données socio-économiques: {e}")
        # Créer un DataFrame vide en cas d'erreur
        return pd.DataFrame()


# Fonction pour obtenir les données socio-économiques d'un IRIS
def get_socioeconomic_data_for_iris(iris_code, socioeconomic_data):
    """
    Récupère les données socio-économiques pour un code IRIS donné
    """
    try:
        # Filtrer les données pour l'IRIS spécifié
        iris_data = socioeconomic_data[socioeconomic_data["IRIS"] == iris_code]

        if not iris_data.empty:
            return {
                "DEC_MED21": iris_data["DEC_MED21"].values[0],
                "DEC_D121": iris_data["DEC_D121"].values[0],
                "DEC_D921": iris_data["DEC_D921"].values[0],
                "DEC_GI21": iris_data["DEC_GI21"].values[0]
            }
        return None
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données socio-économiques: {e}")
        return None


# Titre de l'application
st.title("🏠 Prédiction du Prix Immobilier")

# Sidebar pour le chargement du modèle
with st.sidebar:
    st.title("Configuration")

    if st.button("Charger le modèle"):
        with st.spinner("Chargement du modèle..."):
            try:
                # Charger le modèle scikit-learn
                model_tuple = load_sklearn_model()

                # Vérifier si le modèle a été chargé correctement
                if model_tuple[0] is None:
                    st.error("Échec du chargement du modèle. Le modèle est None.")
                else:
                    # Stocker le modèle, préprocesseur et le chemin temporaire dans le state de la session
                    st.session_state["model"] = model_tuple[0]  # Modèle
                    st.session_state["preprocessor"] = model_tuple[1]  # Préprocesseur
                    st.session_state["temp_dir"] = model_tuple[2]  # Chemin temporaire

                    # Afficher le type du modèle pour confirmation
                    st.success(f"Modèle chargé avec succès! Type: {type(st.session_state['model']).__name__}")
            except Exception as e:
                st.error(f"Erreur lors du chargement du modèle: {e}")
                import traceback
                st.error(traceback.format_exc())

# Conteneur principal pour le formulaire
main_container = st.container()

with main_container:
    # Interface en deux colonnes
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Caractéristiques du bien")

        # Formulaire pour les caractéristiques du bien
        with st.form(key="property_form"):
            # Informations temporelles

            # Caractéristiques du bien
            col_a, col_b = st.columns(2)
            with col_a:
                surface_reelle_bati = st.number_input("Surface habitable (m²)", min_value=10, max_value=500, value=40)
            with col_b:
                nombre_pieces_principales = st.number_input("Nombre de pièces", min_value=1, max_value=15, value=2)

            col_a, col_b = st.columns(2)
            with col_a:
                surface_terrain = st.number_input("Surface terrain (m²)", min_value=0, max_value=10000, value=0)
            with col_b:
                ratio_terrain_bati = surface_terrain / surface_reelle_bati if surface_reelle_bati > 0 else 0
                st.write(f"Ratio terrain/bâti: {ratio_terrain_bati:.2f}")

            # Type de bien
            code_type_local = st.selectbox(
                "Type de bien",
                options=[(1, "Maison"), (2, "Appartement")],
                format_func=lambda x: x[1],
                index=1
            )[0]  # On récupère le code

            # Adresse pour géolocalisation
            address = st.text_input("Adresse complète", "10 rue Jehan de Bazvalan, 56000 Vannes")

            # Bouton de soumission
            submit_button = st.form_submit_button(label="Estimer le prix")

    # Traitement après soumission du formulaire
    if submit_button:
        with st.spinner("Estimation en cours..."):
            # Vérifier si le modèle est chargé
            if "model" not in st.session_state or "preprocessor" not in st.session_state:
                st.error("Veuillez d'abord charger le modèle!")
            else:
                # Géolocalisation de l'adresse
                location_info = get_coordinates_from_address(address)

                if location_info:
                    # Afficher les informations de localisation
                    with col2:
                        st.header("Localisation")
                        st.write(
                            f"**Adresse**: {location_info['street']}, {location_info['postcode']} {location_info['city']}")
                        st.write(f"**Coordonnées**: Lat: {location_info['lat']}, Lon: {location_info['lon']}")

                        # Créer une carte pour afficher la position
                        map_data = pd.DataFrame({
                            'lat': [location_info['lat']],
                            'lon': [location_info['lon']]
                        })
                        st.map(map_data)

                    # Charger les contours IRIS
                    with st.spinner("Recherche du code IRIS..."):
                        iris_contours = load_iris_contours()

                        if not iris_contours.empty:
                            # Trouver le code IRIS
                            iris_info = get_iris_code_from_coordinates(
                                location_info['lat'],
                                location_info['lon'],
                                iris_contours
                            )

                            if iris_info:
                                # Afficher les informations IRIS
                                with col2:
                                    st.subheader("Informations de la zone")
                                    st.write(f"**Code IRIS**: {iris_info['iris_code']}")
                                    st.write(f"**Nom IRIS**: {iris_info['iris_name']}")
                                    st.write(f"**Commune**: {iris_info['commune']}")

                                # Charger les données socio-économiques
                                socioeconomic_data = load_socioeconomic_data()

                                if not socioeconomic_data.empty:
                                    # Récupérer les données socio-économiques de cet IRIS
                                    iris_socioeco = get_socioeconomic_data_for_iris(
                                        iris_info['iris_code'],
                                        socioeconomic_data
                                    )

                                    if iris_socioeco:
                                        for key in ['DEC_MED21', 'DEC_D121', 'DEC_D921', 'DEC_GI21']:
                                            if isinstance(iris_socioeco[key], str):
                                                # Remplacer la virgule par un point et convertir en float
                                                try:
                                                    # Afficher la valeur pour le débogage
                                                    print(f"Valeur d'origine pour {key}: {iris_socioeco[key]}")
                                                    iris_socioeco[key] = float(
                                                        str(iris_socioeco[key]).replace(',', '.'))
                                                    print(f"Valeur convertie pour {key}: {iris_socioeco[key]}")
                                                except Exception as e:
                                                    print(f"Erreur lors de la conversion de {key}: {e}")
                                                    # Valeur par défaut en cas d'erreur
                                                    iris_socioeco[key] = 0.0
                                        # Afficher les données socio-économiques
                                        with col2:
                                            st.subheader("Données socio-économiques")
                                            st.write(f"**Revenu médian**: {iris_socioeco['DEC_MED21']} €")
                                            st.write(f"**1er décile**: {iris_socioeco['DEC_D121']} €")
                                            st.write(f"**9ème décile**: {iris_socioeco['DEC_D921']} €")
                                            st.write(f"**Indice de Gini**: {iris_socioeco['DEC_GI21']}")

                                        current_date = datetime.now()
                                        current_year = current_date.year
                                        current_month = current_date.month

                                        # Préparer les caractéristiques pour la prédiction selon le format attendu par le modèle
                                        features = {
                                            "annee_mutation": current_year,
                                            "mois_mutation": current_month,
                                            "code_postal": location_info['postcode'],
                                            "surface_reelle_bati": surface_reelle_bati,
                                            "nombre_pieces_principales": nombre_pieces_principales,
                                            "surface_terrain": surface_terrain,
                                            "longitude": location_info['lon'],
                                            "latitude": location_info['lat'],
                                            "code_type_local": code_type_local,
                                            "DEC_MED21": iris_socioeco['DEC_MED21'],
                                            "DEC_D121": iris_socioeco['DEC_D121'],
                                            "DEC_D921": iris_socioeco['DEC_D921'],
                                            "DEC_GI21": iris_socioeco['DEC_GI21'],
                                            "ratio_terrain_bati": ratio_terrain_bati,
                                        }

                                        # Faire la prédiction avec le modèle scikit-learn
                                        try:
                                            predicted_price = predict_price_sklearn(
                                                st.session_state["model"],
                                                st.session_state["preprocessor"],
                                                features
                                            )

                                            # Afficher le résultat
                                            st.header("Résultat de l'estimation")
                                            st.success(f"Prix estimé: **{predicted_price:,.0f} €**")

                                            # Calculer le prix au m²
                                            price_per_sqm = predicted_price / surface_reelle_bati
                                            st.info(f"Prix au m²: **{price_per_sqm:,.0f} €/m²**")

                                        except Exception as e:
                                            st.error(f"Erreur lors de la prédiction: {e}")
                                            st.error(f"Détails: {str(e)}")
                                    else:
                                        st.warning("Aucune donnée socio-économique trouvée pour cet IRIS.")
                                else:
                                    st.warning("Impossible de charger les données socio-économiques.")
                            else:
                                st.warning("Aucun code IRIS trouvé pour cette localisation.")
                        else:
                            st.warning("Impossible de charger les contours IRIS.")
                else:
                    st.error("Impossible de géolocaliser cette adresse.")

# Ajouter des informations en bas de page
st.markdown("---")
st.caption("Application de prédiction de prix immobilier basée sur un modèle scikit-learn")