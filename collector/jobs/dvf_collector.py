import requests
from io import BytesIO
import json
import logging
import gzip
import tempfile
import shutil
from typing import Dict, Any, List

from collector.common.base_collector_job import BaseCollectorJob


class DVFCollectorJob(BaseCollectorJob):
    """
    Collecteur pour les données DVF (Demande de Valeurs Foncières).
    """

    def __init__(
            self,
            years: List[str] = None,
            departments: List[str] = None,
            dvf_url: str = "https://files.data.gouv.fr/geo-dvf/latest/csv",
            **kwargs
    ):
        """
        Initialise un collecteur de données DVF.

        Args :
            years : Liste des années à collecter (par défaut : ["2019", "2020", "2021", "2022", "2023", "2024"])
            departments : Liste des départements à collecter (par défaut : ["22", "29", "35", "56"])
            dvf_url : URL de base pour les données DVF
        """
        super().__init__(
            app_name="DVF Data Collector",
            bucket_name="raw-data",
            **kwargs
        )

        self.years = years or ["2019", "2020", "2021", "2022", "2023", "2024"]
        self.departments = departments or ["22", "29", "35", "56"]
        self.dvf_url = dvf_url
        self.temp_dir = None

    def _collect_data(self, **kwargs) -> Dict[str, Any]:
        """
        Télécharge les données DVF pour les départements et années spécifiés.

        Returns :
            Dictionnaire contenant les données collectées
        """
        self.logger.info(f"Collecte des données DVF pour les départements: {self.departments}, années: {self.years}")

        # Créer un répertoire temporaire pour stocker les fichiers décompressés
        self.temp_dir = tempfile.mkdtemp()

        failed_downloads = []
        successful_downloads = []

        for year in self.years:
            for dept in self.departments:
                # URL correcte pour les fichiers DVF
                file_url = f"{self.dvf_url}/{year}/departements/{dept}.csv.gz"

                try:
                    self.logger.info(
                        f"Téléchargement des données DVF pour le département {dept}, année {year} - URL: {file_url}"
                    )

                    # Tentative de téléchargement
                    response = requests.get(file_url, stream=True)

                    if response.status_code != 200:
                        error_msg = f"Erreur lors du téléchargement: {response.status_code} - {file_url}"
                        self.logger.error(error_msg)
                        failed_downloads.append({
                            'url': file_url,
                            'status_code': response.status_code,
                            'departement': dept,
                            'annee': year
                        })
                        continue

                    successful_downloads.append({
                        'content': response.content,
                        'departement': dept,
                        'annee': year,
                        'url': file_url
                    })

                except Exception as e:
                    error_msg = f"Erreur lors du téléchargement pour {dept}, {year}: {str(e)}"
                    self.logger.error(error_msg)
                    failed_downloads.append({
                        'url': file_url,
                        'error': str(e),
                        'departement': dept,
                        'annee': year
                    })

        return {
            'successful_downloads': successful_downloads,
            'failed_downloads': failed_downloads,
            'departments': self.departments,
            'years': self.years,
            'total_success': len(successful_downloads),
            'total_failures': len(failed_downloads)
        }

    def _process_data(self, collected_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Traite les données collectées en décompressant les fichiers gzip.

        Args :
            collected_data : Résultat de la collecte

        Returns :
            Données traitées
        """
        self.logger.info("Traitement des données DVF")

        successful_downloads = collected_data.get('successful_downloads', [])
        processed_files = []

        for download in successful_downloads:
            dept = download.get('departement')
            year = download.get('annee')
            content = download.get('content')
            file_url = download.get('url')

            try:
                # Décompression du fichier
                self.logger.info(f"Décompression du fichier pour le département {dept}, année {year}")

                # Décompresser les données en mémoire
                compressed_data = BytesIO(content)
                decompressed_data = gzip.GzipFile(fileobj=compressed_data, mode='rb')
                csv_content = decompressed_data.read()

                processed_files.append({
                    'departement': dept,
                    'annee': year,
                    'url_source': file_url,
                    'content': csv_content,
                    'size': len(csv_content),
                    'decompressed': True,
                    'format': 'csv'
                })

                self.logger.info(f"Fichier pour le département {dept}, année {year} décompressé avec succès")

            except Exception as e:
                error_msg = f"Erreur lors de la décompression pour {dept}, {year}: {str(e)}"
                self.logger.warning(error_msg)

                # En cas d'erreur, on garde le fichier compressé
                processed_files.append({
                    'departement': dept,
                    'annee': year,
                    'url_source': file_url,
                    'content': content,
                    'size': len(content),
                    'decompressed': False,
                    'format': 'csv.gz',
                    'decompression_error': str(e)
                })

                self.logger.info(f"Fichier compressé pour le département {dept}, année {year} conservé")

        return {
            'processed_files': processed_files,
            'failed_downloads': collected_data.get('failed_downloads', []),
            'departments': collected_data.get('departments', []),
            'years': collected_data.get('years', []),
            'total_processed': len(processed_files)
        }

    def _save_data(self, processed_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Sauvegarde les données traitées avec une structure par dossiers pour le versionnement.

        Structure :
            dvf-data/
              └── YYYYMMDD/
                  ├── dvf_DD_YYYY.csv (ou .csv.gz)
                  └── metadata.json
                  └── failed_downloads.json (si nécessaire)

        Args :
            processed_data : Résultat du traitement

        Returns :
            Informations sur les fichiers sauvegardés
        """
        self.logger.info("Sauvegarde des données DVF avec structure par dossiers")

        # Format de date pour le dossier
        folder_date = self.execution_date.strftime("%Y%m%d")
        folder_path = f"{folder_date}/"

        processed_files = processed_data.get('processed_files', [])
        failed_downloads = processed_data.get('failed_downloads', [])

        saved_files = []

        # Sauvegarde des fichiers traités
        for file_info in processed_files:
            dept = file_info.get('departement')
            year = file_info.get('annee')
            content = file_info.get('content')
            decompressed = file_info.get('decompressed', False)
            format_ext = file_info.get('format')

            # Nom du fichier avec le chemin du dossier
            object_name = f"dvf-data/{folder_path}dvf_{dept}_{year}.{format_ext}"

            # Type de contenu selon le format
            content_type = 'text/csv' if decompressed else 'application/gzip'

            # Sauvegarde du fichier
            self.put_object(
                object_name=object_name,
                data=BytesIO(content),
                content_type=content_type
            )

            saved_files.append({
                'file_name': object_name,
                'departement': dept,
                'annee': year,
                'format': format_ext,
                'taille': len(content),
                'decompressed': decompressed
            })

            self.logger.info(f"Fichier {object_name} sauvegardé avec succès")

        # Sauvegarde des informations sur les échecs de téléchargement
        if failed_downloads:
            failures_object_name = f"{folder_path}failed_downloads.json"
            failures_json = json.dumps(failed_downloads, indent=2)

            self.put_object(
                object_name=failures_object_name,
                data=BytesIO(failures_json.encode()),
                content_type='application/json'
            )

            self.logger.warning(
                f"{len(failed_downloads)} téléchargements ont échoué. Voir {failures_object_name} pour les détails."
            )

        # Création d'un fichier de métadonnées
        metadata = {
            'date_extraction': self.execution_date.isoformat(),
            'departements': processed_data.get('departments', []),
            'annees': processed_data.get('years', []),
            'fichiers': saved_files,
            'echecs': len(failed_downloads)
        }

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
            'failed_count': len(failed_downloads),
            'total_files': len(saved_files)
        }

    def _cleanup(self, **kwargs):
        """Nettoie les ressources temporaires"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                self.logger.info(f"Répertoire temporaire nettoyé: {self.temp_dir}")
                self.temp_dir = None
            except Exception as e:
                self.logger.warning(f"Erreur lors du nettoyage du répertoire temporaire: {str(e)}")
        return {"status": "cleanup_complete"}


if __name__ == "__main__":
    import os
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
        # Récupération des années et départements depuis les arguments ou utilisation des valeurs par défaut
        years = sys.argv[1:] if len(sys.argv) > 1 else None  # Utilise les valeurs par défaut si non spécifié
        departments = None  # Utilise les valeurs par défaut (départements bretons)

        if years:
            print(f"Démarrage de la collecte DVF pour les années: {years}")
        else:
            print("Démarrage de la collecte DVF avec les années par défaut")

        # Création du collecteur (avec minio pointant vers localhost pour les tests)
        collector = DVFCollectorJob(
            years=years,
            departments=departments,
            minio_endpoint="localhost:9000"
        )

        # Exécution du collecteur
        result = collector.run()

        # Affichage du résultat
        if result.get('status') == 'success':
            print("\n✅ Collecte terminée avec succès!")

            # Accès aux résultats des différentes étapes
            save_result = result.get('results', {}).get('save', {})
            print(f"Nombre de fichiers sauvegardés: {save_result.get('total_files', 0)}")
            print(f"Nombre d'échecs: {save_result.get('failed_count', 0)}")

            # Liste des fichiers générés
            if len(save_result.get('files', [])) > 10:
                print(f"\n{len(save_result.get('files', []))} fichiers générés (affichage des 10 premiers):")
                for file in save_result.get('files', [])[:10]:
                    print(f"- {file.get('file_name')} ({file.get('taille')} octets)")
                print("...")
            else:
                print("\nFichiers générés:")
                for file in save_result.get('files', []):
                    print(f"- {file.get('file_name')} ({file.get('taille')} octets)")

            print(f"\nFichier de métadonnées: {save_result.get('metadata_file')}")
        else:
            print(f"\n❌ Erreur lors de la collecte: {result.get('error', 'Erreur inconnue')}")

    except Exception as e:
        print(f"\n❌ Exception lors de l'exécution: {str(e)}")
    finally:
        print("\nFin de l'exécution.")