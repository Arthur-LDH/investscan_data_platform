import os
import tempfile
import joblib
from minio import Minio
import pandas as pd
import numpy as np


def load_sklearn_model():
    """
    Charge le modèle scikit-learn depuis MinIO

    Returns:
        tuple: (modèle, préprocesseur, dossier temporaire)
    """
    # Configuration du client MinIO
    minio_client = Minio(
        endpoint="minio:9000",  # Ajustez selon votre configuration pour Streamlit
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False
    )

    # Créer un dossier temporaire
    temp_dir = tempfile.mkdtemp()
    print(f"Dossier temporaire créé: {temp_dir}")

    try:
        # Paramètres du bucket et du modèle
        bucket_name = "models"
        model_folder = "immobilier_value_predictor_model"  # Dossier fixe sans version

        # Fichiers attendus
        model_filename = "immobilier_model.joblib"
        preprocessor_filename = "immobilier_preprocessor.joblib"

        # Télécharger le modèle
        model_path = os.path.join(temp_dir, model_filename)
        minio_client.fget_object(
            bucket_name,
            f"{model_folder}/{model_filename}",
            model_path
        )
        print(f"Modèle téléchargé: {model_path}")

        # Télécharger le préprocesseur
        preprocessor_path = os.path.join(temp_dir, preprocessor_filename)
        minio_client.fget_object(
            bucket_name,
            f"{model_folder}/{preprocessor_filename}",
            preprocessor_path
        )
        print(f"Préprocesseur téléchargé: {preprocessor_path}")

        # Charger le modèle et le préprocesseur
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)

        print(f"Modèle chargé avec succès! Type: {type(model).__name__}")

        return model, preprocessor, temp_dir

    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        import traceback
        print(traceback.format_exc())
        import shutil
        shutil.rmtree(temp_dir)  # Nettoyer le dossier temporaire
        raise e


def predict_price_sklearn(model, preprocessor, features_dict):
    """
    Fait une prédiction avec le modèle scikit-learn

    Args:
        model: Modèle scikit-learn entraîné
        preprocessor: Préprocesseur associé au modèle
        features_dict: Dictionnaire des caractéristiques du bien

    Returns:
        float: Prix prédit
    """
    try:
        # Convertir le dictionnaire en DataFrame
        features_df = pd.DataFrame([features_dict])

        # S'assurer que toutes les colonnes nécessaires sont présentes
        required_columns = [
            "annee_mutation", "mois_mutation",
            "code_postal", "surface_reelle_bati",
            "nombre_pieces_principales", "surface_terrain",
            "longitude", "latitude", "code_type_local",
            "DEC_MED21", "DEC_D121", "DEC_D921", "DEC_GI21",
            "ratio_terrain_bati"
        ]

        # Compléter les colonnes manquantes
        for col in required_columns:
            if col not in features_df.columns:
                if col in ["annee_mutation", "mois_mutation", "nombre_pieces_principales", "code_type_local"]:
                    features_df[col] = 0
                elif col in ["surface_reelle_bati", "surface_terrain", "longitude", "latitude", "ratio_terrain_bati",
                             "DEC_MED21", "DEC_D121", "DEC_D921", "DEC_GI21"]:
                    features_df[col] = 0.0
                else:
                    features_df[col] = "0"

        # Convertir les types de données si nécessaire
        for col in ["DEC_MED21", "DEC_D121", "DEC_D921", "DEC_GI21"]:
            if col in features_df.columns:
                features_df[col] = features_df[col].astype(float)

        # Convertir code_postal en string (pour le one-hot encoding)
        if "code_postal" in features_df.columns:
            features_df["code_postal"] = features_df["code_postal"].astype(str)

        # Appliquer le préprocesseur
        X_preprocessed = preprocessor.transform(features_df)

        # Faire la prédiction
        predicted_price = model.predict(X_preprocessed)[0]

        return predicted_price

    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        import traceback
        print(traceback.format_exc())
        raise e