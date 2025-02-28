import geopandas as gpd
from io import BytesIO
import requests
import json
import tempfile
import os
import shutil
import subprocess
from typing import Dict, Any, List

from collector.common.base_collector_job import BaseCollectorJob


class BaseTDFiloIRISCollector(BaseCollectorJob):
    """
    Collecteur pour les données des revenus, pauvreté et niveau de vie.
    """

    def __init__(
            self,
            url: str = "https://www.insee.fr/fr/statistiques/fichier/8229323/BASE_TD_FILO_IRIS_2021_DEC_CSV.zip",
            **kwargs
    ):
        """
        Initialise un collecteur de contours géographiques IRIS.

        Args :
            url : URL du fichier
        """
        super().__init__(
            app_name="Base TD FILO IRIS",
            bucket_name="raw-data",
            **kwargs
        )

        self.url = url
        self.temp_dir = None

    def _collect_data(self, **kwargs) -> Dict[str, Any]:
        """
        Télécharge et extrait les données de contours IRIS.

        Returns :
            Dictionnaire contenant les données collectées
        """
        self.logger.info(f"Téléchargement des contours IRIS depuis {self.url}")

        # Création du répertoire temporaire
        self.temp_dir = tempfile.mkdtemp()
        extract_dir = os.path.join(self.temp_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)

        try:
            # Téléchargement
            response = requests.get(self.url, stream=True)

            if response.status_code != 200:
                error_msg = f"Erreur de téléchargement: {response.status_code}"
                raise Exception(error_msg)

            # Enregistrement du fichier
            zip_path = os.path.join(self.temp_dir, "contours.7z")
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)

            file_size = os.path.getsize(zip_path)
            self.logger.info(f"Téléchargement OK: {file_size} octets")

            # Extraction avec 7z
            self.logger.info(f"Extraction vers {extract_dir}...")
            process = subprocess.run(
                ['7z', 'x', zip_path, f'-o{extract_dir}', '-y'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if process.returncode != 0:
                error_msg = f"Erreur d'extraction: {process.stderr}"
                raise Exception(error_msg)

            # Liste des fichiers extraits
            all_files = []
            for root, _, files in os.walk(extract_dir):
                for file in files:
                    all_files.append(os.path.join(root, file))

            self.logger.info(f"Extraction OK: {len(all_files)} fichiers")

            # Recherche de fichiers .csv
            source_file = None
            source_format = None

            for file_path in all_files:
                if file_path.lower().endswith('.csv'):
                    source_file = file_path
                    source_format = 'csv'
                    break

            if not source_file:
                error_msg = "Aucun fichier csv trouvé"
                raise Exception(error_msg)

            self.logger.info(f"Fichier trouvé: {source_file} (format: {source_format})")

            return {
                'source_file': source_file,
                'source_format': source_format,
                'source_url': self.url,
                'all_files': all_files
            }

        except Exception as e:
            self.logger.error(f"Erreur lors de la collecte: {str(e)}")
            # Nettoyage en cas d'erreur
            self._cleanup()
            raise e

    def _process_data(self, collected_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Traite les données collectées.

        Args :
            collected_data : Résultat de la collecte

        Returns :
            Données traitées
        """
        self.logger.info("Traitement des données de contours IRIS")

        source_file = collected_data.get('source_file')
        source_format = collected_data.get('source_format')

        try:
            # On charge temporairement le fichier avec GeoPandas juste pour obtenir des informations
            self.logger.info(f"Lecture des métadonnées de {source_file}...")
            file = gpd.read_file(source_file)

            # Log des informations sur les données
            entity_count = len(file)
            columns = file.columns.tolist()
            self.logger.info(f"Fichier chargé: {entity_count} entités")
            self.logger.info(f"Colonnes disponibles: {columns}")

            # On n'utilise pas le GeoDataFrame pour la suite, on va stocker le fichier original

            # Lire le fichier source en binaire
            with open(source_file, 'rb') as f:
                source_data = f.read()

            file_size = len(source_data)
            self.logger.info(f"Taille du fichier source {source_format}: {file_size} octets")

            return {
                'source_data': source_data,
                'source_format': source_format,
                'source_url': collected_data.get('source_url'),
                'total_entities': entity_count,
                'columns': columns
            }

        except Exception as e:
            self.logger.error(f"Erreur lors du traitement: {str(e)}")
            raise e

    def _save_data(self, processed_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Sauvegarde les données traitées avec une structure par dossiers pour le versionnement.

        Structure :
            base-td-file-iris/
              └── YYYYMMDD/
                  ├── base_td_file_iris.[csv]
                  └── metadata.json

        Args :
            processed_data : Résultat du traitement

        Returns :
            Informations sur les fichiers sauvegardés
        """
        self.logger.info("Sauvegarde du fichier base TD FILO IRIS")

        # Récupération des données
        source_data = processed_data.get('source_data')
        source_format = processed_data.get('source_format')

        if not source_data or not source_format:
            raise ValueError("Données sources manquantes ou format non spécifié")

        # Format de date pour le dossier
        folder_date = self.execution_date.strftime("%Y%m%d")
        folder_path = f"base-td-file-iris/{folder_date}/"

        saved_files = []

        # Déterminer le type MIME selon le format
        content_type = 'application/geopackage+sqlite3' if source_format == 'csv' else 'application/x-esri-shape'

        # Chemin du fichier incluant le dossier daté
        file_name = f"base_td_file_iris.{source_format}"
        object_name = f"{folder_path}{file_name}"

        # Sauvegarde du fichier dans son format d'origine
        self.put_object(
            object_name=object_name,
            data=BytesIO(source_data),
            content_type=content_type
        )

        saved_files.append({
            'file_name': object_name,
            'format': source_format,
            'scope': 'France entière',
            'size_bytes': len(source_data)
        })

        # Création d'un fichier de métadonnées dans le même dossier
        metadata = {
            'date_extraction': self.execution_date.isoformat(),
            'source_url': processed_data.get('source_url'),
            'nombre_entites': processed_data.get('total_entities', 0),
            'colonnes': processed_data.get('columns', []),
            'format': source_format,
            'fichiers': saved_files
        }

        # Le fichier de métadonnées va également dans le dossier daté
        metadata_object_name = f"{folder_path}metadata.json"
        metadata_buffer = BytesIO(json.dumps(metadata, indent=2).encode())

        self.put_object(
            object_name=metadata_object_name,
            data=metadata_buffer,
            content_type='application/json'
        )

        return {
            'folder': folder_date,
            'files': saved_files,
            'metadata_file': metadata_object_name,
            'entity_count': processed_data.get('total_entities', 0),
            'format': source_format
        }

    def _cleanup(self):
        """Nettoie le répertoire temporaire s'il existe"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                self.logger.info(f"Répertoire temporaire nettoyé: {self.temp_dir}")
                self.temp_dir = None
            except Exception as e:
                self.logger.warning(f"Erreur lors du nettoyage du répertoire temporaire: {str(e)}")


if __name__ == "__main__":
    import logging
    import sys

    # Configuration du logging pour afficher les informations
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Création d'une instance du collecteur
    collector = None
    try:
        print("Démarrage de la collecte des contours IRIS pour la France entière")

        # Création du collecteur (avec minio pointant vers localhost pour les tests)
        collector = BaseTDFiloIRISCollector(
            minio_endpoint="localhost:9000"
        )

        # Exécution du collecteur
        result = collector.run()

        # Affichage du résultat
        if result.get('status') == 'success':
            print("\n✅ Collecte terminée avec succès!")

            # Accès aux résultats des différentes étapes
            save_result = result.get('results', {}).get('save', {})
            print(f"Nombre d'entités collectées: {save_result.get('entity_count', 0)}")
            print(f"Format: {save_result.get('format')}")

            # Liste des fichiers générés
            print("\nFichiers générés:")
            for file in save_result.get('files', []):
                print(f"- {file.get('file_name')} ({file.get('size_bytes', 0)} octets)")

            print(f"\nFichier de métadonnées: {save_result.get('metadata_file')}")
        else:
            print(f"\n❌ Erreur lors de la collecte: {result.get('error', 'Erreur inconnue')}")

    except Exception as e:
        print(f"\n❌ Exception lors de l'exécution: {str(e)}")
    finally:
        print("\nFin de l'exécution.")