import logging
import shutil
from pathlib import Path
from typing import Optional
from util.dataset_paths import assets_datasets_dir, assets_uploads_dir
logger = logging.getLogger(__name__)

def cleanup_client_data(client_id: str, preserve_uploads: bool=False, dataset_id: Optional[str]=None) -> None:
    try:
        if dataset_id:
            scoped_xml = Path('xml_prompts/clients') / client_id / 'data_sources' / dataset_id
            if scoped_xml.exists():
                logger.info('Removing XML data sources for client %s dataset %s', client_id, dataset_id)
                shutil.rmtree(scoped_xml)
            ds_dir = assets_datasets_dir(client_id, dataset_id)
            if ds_dir.exists():
                logger.info('Removing datasets for client %s dataset %s', client_id, dataset_id)
                shutil.rmtree(ds_dir)
            if not preserve_uploads:
                up_dir = assets_uploads_dir(client_id, dataset_id)
                if up_dir.exists():
                    logger.info('Removing uploads for client %s dataset %s', client_id, dataset_id)
                    shutil.rmtree(up_dir)
            return
        xml_data_sources_dir = Path('xml_prompts/clients') / client_id / 'data_sources'
        if xml_data_sources_dir.exists():
            logger.info('Removing existing XML data sources for client %s: %s', client_id, xml_data_sources_dir)
            shutil.rmtree(xml_data_sources_dir)
            logger.info('Removed XML data sources directory for client %s', client_id)
        assets_datasets_dir_legacy = Path(f'assets/clients/{client_id}/datasets')
        if assets_datasets_dir_legacy.exists():
            logger.info('Removing existing datasets for client %s: %s', client_id, assets_datasets_dir_legacy)
            shutil.rmtree(assets_datasets_dir_legacy)
            logger.info('Removed datasets directory for client %s', client_id)
        if not preserve_uploads:
            assets_uploads_dir_legacy = Path(f'assets/clients/{client_id}/uploads')
            if assets_uploads_dir_legacy.exists():
                logger.info('Removing existing uploads for client %s: %s', client_id, assets_uploads_dir_legacy)
                shutil.rmtree(assets_uploads_dir_legacy)
                logger.info('Removed uploads directory for client %s', client_id)
        else:
            logger.info('Preserving uploads directory for client %s (file upload processing)', client_id)
    except Exception as e:
        logger.error('Error cleaning up existing data for client %s: %s', client_id, e, exc_info=True)