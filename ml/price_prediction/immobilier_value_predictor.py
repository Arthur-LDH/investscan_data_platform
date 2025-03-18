"""
Modèle de prédiction de prix immobiliers avec scikit-learn
Ce script charge les données, prépare les caractéristiques,
entraîne un modèle GradientBoosting et évalue ses performances.
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


try:
    from minio import Minio
    import io
    import json
except ImportError:
    print("Pour accéder aux données Minio, installez les bibliothèques requises:")
    print("pip install minio")


def load_data(bucket_name="ml-datasets", folder_path="house_price_model", version="latest"):
    try:
        from minio import Minio
        import json
        import io
        import tempfile
        import os
        import shutil

        print("Connexion à Minio...")

        # Configuration du client Minio
        minio_client = Minio(
            endpoint="localhost:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
            secure=False
        )

        # Si la version est 'latest', lire le fichier info.json pour obtenir la version actuelle
        if version == "latest":
            print("Détermination de la dernière version...")
            try:
                # Récupérer le fichier info.json
                info_obj = minio_client.get_object(bucket_name, f"{folder_path}/latest/info.json")
                info_content = info_obj.read()
                info_data = json.loads(info_content.decode('utf-8'))
                version = info_data.get('redirect_to', version)
                print(f"Version actuelle: {version}")
            except Exception as e:
                print(f"Erreur lors de la lecture du fichier info.json: {e}")
                print("Utilisation de la version 'latest'")

        # Construire les chemins des répertoires
        base_path = f"{folder_path}/{version}"
        print('base_path', base_path)
        train_path = f"{base_path}/train.parquet"
        print('train_path', train_path)
        validation_path = f"{base_path}/validation.parquet"
        test_path = f"{base_path}/test.parquet"

        print(f"Chargement des données depuis {bucket_name}/{base_path}...")

        # Fonction pour charger un DataFrame depuis un répertoire parquet dans Minio
        def read_parquet_from_minio(directory_path):
            try:
                # Créer un répertoire temporaire pour télécharger les fichiers parquet
                temp_dir = tempfile.mkdtemp()
                local_path = os.path.join(temp_dir, "parquet_data")
                os.makedirs(local_path, exist_ok=True)

                print(f"Téléchargement du répertoire parquet {directory_path} vers {local_path}...")

                # Lister tous les objets dans le répertoire parquet
                objects = minio_client.list_objects(bucket_name, prefix=directory_path, recursive=True)

                # Télécharger chaque fichier
                for obj in objects:
                    # Ignorer les fichiers cachés et métadonnées
                    if obj.object_name.endswith(".crc") or "/_" in obj.object_name:
                        continue

                    # Déterminer le chemin de fichier local
                    relative_path = obj.object_name[len(directory_path):].lstrip('/')
                    if not relative_path:  # Ignorer le répertoire lui-même
                        continue

                    local_file_path = os.path.join(local_path, relative_path)
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

                    # Télécharger le fichier
                    print(f"Téléchargement de {obj.object_name}...")
                    minio_client.fget_object(bucket_name, obj.object_name, local_file_path)

                # Lire le répertoire parquet
                print(f"Lecture du parquet depuis {local_path}...")
                df = pd.read_parquet(local_path)

                # Nettoyer le répertoire temporaire
                shutil.rmtree(temp_dir)

                return df
            except Exception as e:
                print(f"Erreur lors de la lecture du répertoire parquet {directory_path}: {e}")
                raise

        # Charger les DataFrame
        df_train = read_parquet_from_minio(train_path)
        df_validation = read_parquet_from_minio(validation_path)
        df_test = read_parquet_from_minio(test_path)

        print(f"Nombre de lignes dans le dataset d'entraînement: {len(df_train)}")
        return df_train, df_validation, df_test

    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        raise  # Propager l'erreur sans créer de données synthétiques


def prepare_features(df_train, df_validation, df_test, target_column='valeur_fonciere'):
    """
    Prépare les caractéristiques pour l'entraînement du modèle

    Args:
        df_train: DataFrame d'entraînement
        df_validation: DataFrame de validation
        df_test: DataFrame de test
        target_column: Nom de la colonne cible

    Returns:
        Tuple (X_train, y_train, X_validation, y_validation, X_test, y_test, preprocessor)
    """
    print("Préparation des caractéristiques...")

    # Identifier les types de colonnes
    categorical_cols = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df_train.select_dtypes(include=['int', 'float']).columns.tolist()

    if target_column in numerical_cols:
        numerical_cols.remove(target_column)

    print(f"Colonnes catégorielles: {categorical_cols}")
    print(f"Colonnes numériques: {numerical_cols}")

    # Créer un préprocesseur pour les données
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'  # Ignorer les colonnes non spécifiées
    )

    # Préparer les variables X et y
    X_train = df_train[numerical_cols + categorical_cols]
    y_train = df_train[target_column]

    X_validation = df_validation[numerical_cols + categorical_cols]
    y_validation = df_validation[target_column]

    X_test = df_test[numerical_cols + categorical_cols]
    y_test = df_test[target_column]

    # Appliquer le préprocesseur
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_validation_preprocessed = preprocessor.transform(X_validation)
    X_test_preprocessed = preprocessor.transform(X_test)

    print(f"Dimensions après prétraitement - X_train: {X_train_preprocessed.shape}")

    return X_train_preprocessed, y_train, X_validation_preprocessed, y_validation, X_test_preprocessed, y_test, preprocessor


def train_model(X_train, y_train):
    """
    Entraîne un modèle de Gradient Boosting optimisé

    Args:
        X_train: Features d'entraînement prétraitées
        y_train: Cible d'entraînement

    Returns:
        Modèle entraîné
    """
    print("Entraînement du modèle Gradient Boosting...")

    # Hyperparamètres optimisés (vous pouvez les ajuster selon vos besoins)
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        verbose=1
    )

    # Entraîner le modèle
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X, y, dataset_name="validation"):
    """
    Évalue les performances du modèle

    Args:
        model: Modèle entraîné
        X: Features prétraitées
        y: Valeurs cibles réelles
        dataset_name: Nom du dataset pour l'affichage

    Returns:
        Dictionnaire des métriques de performance
    """
    print(f"Évaluation du modèle sur le jeu de {dataset_name}...")

    # Générer des prédictions
    y_pred = model.predict(X)

    # Calculer les métriques
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'mae': mean_absolute_error(y, y_pred),
        'r2': r2_score(y, y_pred)
    }

    # Calculer le pourcentage d'erreur
    percent_error = np.abs(y_pred - y) / y * 100
    metrics['mean_percent_error'] = np.mean(percent_error)
    metrics['median_percent_error'] = np.median(percent_error)

    # Afficher les résultats
    print(f"Performances sur le jeu de {dataset_name}:")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"R²: {metrics['r2']:.4f}")
    print(f"Erreur moyenne: {metrics['mean_percent_error']:.2f}%")
    print(f"Erreur médiane: {metrics['median_percent_error']:.2f}%")

    return metrics


def save_model(model, preprocessor, output_dir="./models", save_to_minio=False, bucket_name="models"):
    """
    Sauvegarde le modèle et le préprocesseur en local et optionnellement sur Minio

    Args:
        model: Modèle entraîné
        preprocessor: Préprocesseur
        output_dir: Répertoire de sauvegarde local
        save_to_minio: Booléen indiquant si le modèle doit être sauvegardé sur Minio
        bucket_name: Nom du bucket Minio pour la sauvegarde

    Returns:
        Tuple (chemin du modèle, chemin du préprocesseur)
    """
    # Sauvegarde en local
    print(f"Sauvegarde du modèle en local dans {output_dir}...")

    # Créer le répertoire s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Noms des fichiers
    model_filename = "immobilier_model.joblib"
    preprocessor_filename = "immobilier_preprocessor.joblib"

    # Chemins complets
    model_path = os.path.join(output_dir, model_filename)
    preprocessor_path = os.path.join(output_dir, preprocessor_filename)

    # Sauvegarde locale
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    print(f"Modèle sauvegardé en local: {model_path}")
    print(f"Préprocesseur sauvegardé en local: {preprocessor_path}")

    # Sauvegarde sur Minio si demandé
    if save_to_minio:
        try:
            print(f"Sauvegarde du modèle sur Minio dans le bucket {bucket_name}...")

            # Configuration du client Minio
            minio_client = Minio(
                endpoint="localhost:9000",
                access_key="minioadmin",
                secret_key="minioadmin",
                secure=False
            )

            # Vérifier si le bucket existe, sinon le créer
            if not minio_client.bucket_exists(bucket_name):
                minio_client.make_bucket(bucket_name)
                print(f"Bucket '{bucket_name}' créé")

            # Définir un dossier avec timestamp pour les versions
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            minio_folder = f"immobilier_value_predictor_model"

            # Télécharger le modèle et le préprocesseur
            minio_client.fput_object(bucket_name, f"{minio_folder}/{model_filename}", model_path)
            minio_client.fput_object(bucket_name, f"{minio_folder}/{preprocessor_filename}", preprocessor_path)

            # Création d'un fichier metadata.json
            metadata = {
                "created_at": timestamp,
                "model_type": "GradientBoostingRegressor",
                "description": "Modèle de prédiction des prix immobiliers"
            }

            # Sauvegarder le metadata.json
            with open(os.path.join(output_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f)

            minio_client.fput_object(
                bucket_name,
                f"{minio_folder}/metadata.json",
                os.path.join(output_dir, "metadata.json")
            )

            print(f"Modèle sauvegardé sur Minio: {bucket_name}/{minio_folder}")

        except Exception as e:
            print(f"Erreur lors de la sauvegarde sur Minio: {e}")

    return model_path, preprocessor_path


def main():
    """Fonction principale"""
    # 1. Charger les données depuis Minio
    df_train, df_validation, df_test = load_data(
        bucket_name="ml-datasets",
        folder_path="house_price_model",
        version="latest"
    )

    # 2. Préparer les caractéristiques
    X_train, y_train, X_validation, y_validation, X_test, y_test, preprocessor = prepare_features(
        df_train, df_validation, df_test, target_column='valeur_fonciere'
    )

    # 3. Entraîner le modèle
    model = train_model(X_train, y_train)

    # 4. Évaluer sur l'ensemble de validation
    validation_metrics = evaluate_model(model, X_validation, y_validation, dataset_name="validation")

    # 5. Évaluer sur l'ensemble de test
    test_metrics = evaluate_model(model, X_test, y_test, dataset_name="test")

    # 6. Sauvegarder le modèle et le préprocesseur (localement et sur Minio)
    model_path, preprocessor_path = save_model(
        model,
        preprocessor,
        output_dir="./models",
        save_to_minio=True,
        bucket_name="models"
    )

    print("\nTraitement terminé.")


if __name__ == "__main__":
    main()