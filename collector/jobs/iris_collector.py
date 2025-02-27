import pandas as pd
from io import BytesIO
import requests
import json
from typing import Dict, Any, List

from collector.common.base_collector_job import BaseCollectorJob


class IRISCollectorJob(BaseCollectorJob):
    """
    Collecteur pour les données IRIS depuis OpenDataSoft.
    """

    def __init__(
            self,
            regions: List[str] = None,
            base_url: str = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/georef-france-iris/records",
            limit: int = 100,
            **kwargs
    ):
        """
        Initialise un collecteur de données IRIS.

        Args :
            regions : Liste des régions à collecter
            base_url : URL de l'API
            limit : Nombre d'enregistrements par requête
        """
        super().__init__(
            app_name="IRIS Data Collector",
            bucket_name="raw-data",
            **kwargs
        )

        self.regions = regions
        self.base_url = base_url
        self.limit = limit

    def _collect_data(self, **kwargs) -> Dict[str, Any]:
        """
        Collecte les données IRIS depuis l'API.

        Returns :
            Dictionnaire contenant les données collectées
        """
        self.logger.info(f"Collecte des données IRIS pour les régions: {self.regions}")
        all_data = []

        for region in self.regions:
            offset = 0
            total_records = None

            while total_records is None or offset < total_records:
                # Construction de l'URL avec les paramètres
                params = {
                    'where': f'reg_name = "{region}"',
                    'limit': self.limit,
                    'offset': offset
                }

                # Requête à l'API
                response = requests.get(self.base_url, params=params)

                if response.status_code != 200:
                    raise Exception(f"Erreur lors de la récupération des données IRIS: {response.status_code}")

                data = response.json()

                # Mise à jour du nombre total d'enregistrements si pas encore fait
                if total_records is None:
                    total_records = data['total_count']

                # Ajout des résultats à notre liste
                all_data.extend(data['results'])

                # Mise à jour de l'offset pour la prochaine requête
                offset += self.limit

                self.logger.info(f"Récupérés {len(all_data)}/{total_records} enregistrements pour {region}")

        return {
            'raw_data': all_data,
            'regions': self.regions,
            'total_records': len(all_data)
        }

    def _process_data(self, collected_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Traite les données collectées.

        Args :
            collected_data : Résultat de la collecte

        Returns :
            Données traitées
        """
        self.logger.info("Traitement des données IRIS")

        # Extraction des données brutes
        raw_data = collected_data.get('raw_data', [])

        # Conversion en DataFrame
        df = pd.json_normalize(raw_data)

        self.logger.info(f"Données traitées: {len(df)} enregistrements, {len(df.columns)} colonnes")

        return {
            'dataframe': df,
            'regions': collected_data.get('regions', []),
            'total_records': len(df),
            'columns': df.columns.tolist()
        }

    def _save_data(self, processed_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Sauvegarde les données traitées avec une structure par dossiers pour le versionnement.

        Structure :
            iris-data/
              └── YYYYMMDD/
                  ├── iris_data.csv
                  └── metadata.json

        Args :
            processed_data : Résultat du traitement

        Returns :
            Informations sur les fichiers sauvegardés
        """
        self.logger.info("Sauvegarde des données IRIS en CSV avec structure par dossiers")

        # Récupération du DataFrame
        df = processed_data.get('dataframe')
        if df is None or len(df) == 0:
            raise ValueError("Aucune donnée à sauvegarder")

        # Format de date pour le dossier
        folder_date = self.execution_date.strftime("%Y%m%d")
        folder_path = f"iris-data/{folder_date}/"

        saved_files = []

        # Sauvegarde en CSV uniquement
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        # Chemin du fichier incluant le dossier daté
        csv_object_name = f"{folder_path}iris_data.csv"
        self.put_object(
            object_name=csv_object_name,
            data=csv_buffer,
            content_type='text/csv'
        )
        saved_files.append({
            'file_name': csv_object_name,
            'format': 'csv'
        })

        # Création d'un fichier de métadonnées dans le même dossier
        metadata = {
            'date_generation': self.execution_date.isoformat(),
            'regions': processed_data.get('regions', []),
            'nombre_enregistrements': processed_data.get('total_records', 0),
            'colonnes': processed_data.get('columns', []),
            'source': self.base_url,
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
            'record_count': processed_data.get('total_records', 0)
        }

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
        # Utilise les régions passées en argument ou Bretagne par défaut
        regions = sys.argv[1:] if len(sys.argv) > 1 else ["Bretagne"]

        print(f"Démarrage de la collecte pour les régions: {regions}")

        # Création du collecteur (avec minio pointant vers localhost pour les tests)
        collector = IRISCollectorJob(
            regions=regions,
            minio_endpoint="localhost:9000"  # Généralement localhost pour les tests locaux
        )

        # Exécution du collecteur
        result = collector.run()

        # Affichage du résultat
        if result.get('status') == 'success':
            print("\n✅ Collecte terminée avec succès!")

            # Accès aux résultats des différentes étapes
            save_result = result.get('results', {}).get('save', {})
            print(f"Nombre d'enregistrements collectés: {save_result.get('record_count', 0)}")

            # Liste des fichiers générés
            print("\nFichiers générés:")
            for file in save_result.get('files', []):
                print(f"- {file.get('file_name')}")

            print(f"\nFichier de métadonnées: {save_result.get('metadata_file')}")
        else:
            print(f"\n❌ Erreur lors de la collecte: {result.get('error', 'Erreur inconnue')}")

    except Exception as e:
        print(f"\n❌ Exception lors de l'exécution: {str(e)}")
    finally:
        print("\nFin de l'exécution.")