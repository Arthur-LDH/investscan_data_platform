from abc import ABC, abstractmethod
import json
import logging
from datetime import datetime
from io import BytesIO
from typing import Dict, Any, Optional

from minio import Minio


class BaseCollectorJob(ABC):
    """
    Classe abstraite pour standardiser les jobs de collecte de données.
    Adaptée pour une intégration par étapes avec Airflow.
    """

    def __init__(
            self,
            app_name: str,
            bucket_name: str,
            minio_endpoint: str = "minio:9000",
            minio_access_key: str = "minioadmin",
            minio_secret_key: str = "minioadmin",
            secure: bool = False
    ):
        """
        Initialise un nouveau job de collecte.

        Args:
            app_name: Nom de l'application/collecteur
            bucket_name: Nom du bucket MinIO pour le stockage
            minio_endpoint: Endpoint du serveur MinIO
            minio_access_key: Clé d'accès MinIO
            minio_secret_key: Clé secrète MinIO
            secure: Utiliser HTTPS pour la connexion MinIO
        """
        self.app_name = app_name
        self.bucket_name = bucket_name

        # Configuration MinIO
        self.minio_endpoint = minio_endpoint
        self.minio_access_key = minio_access_key
        self.minio_secret_key = minio_secret_key
        self.secure = secure

        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)

        # Date d'exécution
        self.execution_date = datetime.now()

        # Client MinIO
        self._minio_client = None

        # État global pour faciliter la persistance entre les étapes dans Airflow
        self.state = {}

    @property
    def minio_client(self) -> Minio:
        """
        Lazy initialization du client MinIO.
        """
        if self._minio_client is None:
            self._minio_client = Minio(
                endpoint=self.minio_endpoint,
                access_key=self.minio_access_key,
                secret_key=self.minio_secret_key,
                secure=self.secure
            )
        return self._minio_client

    def ensure_bucket_exists(self) -> bool:
        """
        Vérifie si le bucket existe et le crée si nécessaire.

        Returns :
            True si le bucket existe ou a été créé
        """
        try:
            if not self.minio_client.bucket_exists(self.bucket_name):
                self.minio_client.make_bucket(self.bucket_name)
                self.logger.info(f"Bucket {self.bucket_name} créé avec succès")
            else:
                self.logger.info(f"Bucket {self.bucket_name} existe déjà")
            return True
        except Exception as e:
            self.logger.error(f"Erreur lors de la vérification/création du bucket: {str(e)}")
            raise

    def save_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        Sauvegarde les métadonnées dans le stockage.

        Args :
            metadata : Les métadonnées à sauvegarder

        Returns :
            Le nom du fichier de métadonnées
        """
        # Enrichir les métadonnées avec des informations communes
        full_metadata = {
            'app_name': self.app_name,
            'date_execution': self.execution_date.isoformat(),
            'bucket': self.bucket_name,
            **metadata
        }

        current_date = self.execution_date.strftime("%Y%m%d")
        metadata_object_name = f"{self.bucket_name}_metadata_{current_date}.json"

        metadata_json = json.dumps(full_metadata, indent=2)
        metadata_buffer = BytesIO(metadata_json.encode())

        self.minio_client.put_object(
            bucket_name=self.bucket_name,
            object_name=metadata_object_name,
            data=metadata_buffer,
            length=len(metadata_buffer.getvalue()),
            content_type='application/json'
        )

        self.logger.info(f"Métadonnées sauvegardées: {metadata_object_name}")
        return metadata_object_name

    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> str | None:
        """
        Enregistre une erreur dans les logs et dans MinIO.

        Args :
            error : L'exception qui s'est produite
            context : Contexte supplémentaire à enregistrer

        Returns :
            Le nom du fichier d'erreur
        """
        # Préparer le message d'erreur
        timestamp = self.execution_date.strftime("%Y%m%d_%H%M%S")
        error_object_name = f"error_{self.app_name.lower().replace(' ', '_')}_{timestamp}.txt"

        error_message = f"Erreur dans {self.app_name}\n"
        error_message += f"Date: {self.execution_date.isoformat()}\n"
        error_message += f"Type d'erreur: {type(error).__name__}\n"
        error_message += f"Message: {str(error)}\n"

        if context:
            error_message += "Contexte:\n"
            for key, value in context.items():
                error_message += f"  {key}: {value}\n"

        # Log dans le système de logging
        self.logger.error(error_message)

        # Enregistrer dans MinIO si possible
        try:
            self.ensure_bucket_exists()
            self.minio_client.put_object(
                bucket_name=self.bucket_name,
                object_name=error_object_name,
                data=BytesIO(error_message.encode()),
                length=len(error_message),
                content_type='text/plain'
            )

            self.logger.info(f"Détails de l'erreur sauvegardés: {error_object_name}")
            return error_object_name
        except Exception as save_error:
            self.logger.error(f"Impossible de sauvegarder l'erreur: {str(save_error)}")
            return None

    def put_object(
            self,
            object_name: str,
            data: BytesIO,
            content_type: str = 'application/octet-stream',
            metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Sauvegarde un objet dans MinIO.

        Args :
            object_name : Nom de l'objet
            data : Données à sauvegarder
            content_type : Type MIME du contenu
            metadata : Métadonnées associées à l'objet

        Returns :
            Le nom de l'objet sauvegardé
        """
        # Déterminer la taille des données
        data.seek(0, 2)  # Aller à la fin
        length = data.tell()
        data.seek(0)  # Revenir au début

        # Sauvegarder l'objet
        self.minio_client.put_object(
            bucket_name=self.bucket_name,
            object_name=object_name,
            data=data,
            length=length,
            content_type=content_type,
            metadata=metadata
        )

        self.logger.info(f"Objet sauvegardé: {object_name}")
        return object_name

    # Méthodes pour l'intégration avec Airflow - délibérément rendues publiques

    def prepare(self) -> Dict[str, Any]:
        """
        Prépare l'exécution du job.
        Étape 1 du workflow pour Airflow.

        Returns :
            Informations sur l'étape de préparation
        """
        self.logger.info(f"Préparation du job {self.app_name}")
        result = self._prepare()
        self.state['prepare_result'] = result
        return {'status': 'success', 'result': result}

    def collect_data(self, **kwargs) -> Dict[str, Any]:
        """
        Collecte les données depuis leur source.
        Étape 2 du workflow pour Airflow.

        Returns :
            Informations sur les données collectées
        """
        self.logger.info(f"Collecte des données pour {self.app_name}")
        try:
            result = self._collect_data(**kwargs)
            self.state['collect_result'] = result
            return {'status': 'success', 'result': result}
        except Exception as e:
            self.logger.error(f"Erreur lors de la collecte: {str(e)}")
            error_file = self.log_error(e, {'step': 'collect_data'})
            return {
                'status': 'error',
                'error': str(e),
                'error_file': error_file
            }

    def process_data(self, collected_data: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """
        Traite les données collectées.
        Étape 3 du workflow pour Airflow.

        Args :
            collected_data : Données collectées (si fournies)

        Returns :
            Informations sur les données traitées
        """
        if collected_data is None:
            collected_data = self.state.get('collect_result', {})

        self.logger.info(f"Traitement des données pour {self.app_name}")
        try:
            result = self._process_data(collected_data, **kwargs)
            self.state['process_result'] = result
            return {'status': 'success', 'result': result}
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement: {str(e)}")
            error_file = self.log_error(e, {'step': 'process_data'})
            return {
                'status': 'error',
                'error': str(e),
                'error_file': error_file
            }

    def save_data(self, processed_data: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """
        Sauvegarde les données traitées.
        Étape 4 du workflow pour Airflow.

        Args :
            processed_data : Données traitées (si fournies)

        Returns :
            Informations sur les données sauvegardées
        """
        if processed_data is None:
            processed_data = self.state.get('process_result', {})

        self.logger.info(f"Sauvegarde des données pour {self.app_name}")
        try:
            result = self._save_data(processed_data, **kwargs)
            self.state['save_result'] = result
            return {'status': 'success', 'result': result}
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde: {str(e)}")
            error_file = self.log_error(e, {'step': 'save_data'})
            return {
                'status': 'error',
                'error': str(e),
                'error_file': error_file
            }

    def cleanup(self, **kwargs) -> Dict[str, Any]:
        """
        Nettoie les ressources après l'exécution.
        Étape 5 du workflow pour Airflow.

        Returns :
            Informations sur le nettoyage
        """
        self.logger.info(f"Nettoyage pour {self.app_name}")
        try:
            result = self._cleanup(**kwargs)
            return {'status': 'success', 'result': result}
        except Exception as e:
            self.logger.error(f"Erreur lors du nettoyage: {str(e)}")
            error_file = self.log_error(e, {'step': 'cleanup'})
            return {
                'status': 'error',
                'error': str(e),
                'error_file': error_file
            }

    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Exécute le workflow complet de collecte.

        Cette méthode orchestre le processus complet pour une utilisation
        en dehors d'Airflow ou comme une tâche unique dans Airflow.

        Returns :
            Résultat complet de la collecte
        """
        try:
            self.logger.info(f"Démarrage du job {self.app_name}")

            # Assurer que le bucket existe
            self.ensure_bucket_exists()

            # Exécution des étapes
            prepare_result = self.prepare(**kwargs)
            if prepare_result.get('status') == 'error':
                return prepare_result

            collect_result = self.collect_data(**kwargs)
            if collect_result.get('status') == 'error':
                return collect_result

            process_result = self.process_data(collect_result.get('result'), **kwargs)
            if process_result.get('status') == 'error':
                return process_result

            save_result = self.save_data(process_result.get('result'), **kwargs)
            if save_result.get('status') == 'error':
                return save_result

            cleanup_result = self.cleanup(**kwargs)

            # Résultat final
            final_result = {
                'status': 'success',
                'app_name': self.app_name,
                'date': self.execution_date.isoformat(),
                'results': {
                    'prepare': prepare_result.get('result'),
                    'collect': collect_result.get('result'),
                    'process': process_result.get('result'),
                    'save': save_result.get('result'),
                    'cleanup': cleanup_result.get('result')
                }
            }

            self.logger.info(f"Job {self.app_name} terminé avec succès")
            return final_result

        except Exception as e:
            self.logger.error(f"Erreur dans le job {self.app_name}: {str(e)}")
            error_file = self.log_error(e)
            return {
                'status': 'error',
                'error': str(e),
                'error_file': error_file
            }

    def _prepare(self, **kwargs) -> Any:
        """
        Prépare l'exécution du job.
        À implémenter par les classes enfants si nécessaire.
        """
        return None

    @abstractmethod
    def _collect_data(self, **kwargs) -> Any:
        """
        Collecte les données depuis leur source.
        Doit être implémentée par les classes enfants.
        """
        pass

    def _process_data(self, collected_data: Any, **kwargs) -> Any:
        """
        Traite les données collectées.
        À implémenter par les classes enfants si nécessaire.

        Args :
            collected_data : Les données collectées
        """
        return collected_data

    def _save_data(self, processed_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Sauvegarde les données traitées.
        À implémenter par les classes enfants si nécessaire.

        Args :
            processed_data : Les données traitées
        """
        return {"data": processed_data}

    def _cleanup(self, **kwargs) -> Any:
        """
        Nettoie les ressources après l'exécution.
        À implémenter par les classes enfants si nécessaire.
        """
        return None