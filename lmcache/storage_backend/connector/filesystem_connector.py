import os
from typing import Optional
from lmcache.logging import init_logger
from lmcache.storage_backend.connector.base_connector import \
    RemoteBytesConnector

logger = init_logger(__name__)


class FilesystemConnector(RemoteBytesConnector):

    def __init__(self, path: str):
        self.path = path

    def _key_to_path(self, key: str) -> str:
        return os.path.join(self.path, f"kvcache_{key}")

    def exists(self, key: str) -> bool:
        logger.debug(f"FilesystemConnector.exists({key}) ...")
        return os.path.isfile(self._key_to_path(key))

    def get(self, key: str) -> Optional[bytes]:
        logger.debug(f"FilesystemConnector.get({key}) ...")
        with open(self._key_to_path(key), 'rb') as f:
            return f.read()

    def set(self, key: str, obj: bytes) -> None:
        logger.debug(f"FilesystemConnector.set({key}) ...")
        with open(self._key_to_path(key), 'wb') as f:
            f.write(obj)


    def list(self):
        logger.debug("FilesystemConnector.list() ...")
        entries = os.listdir(self.path)     
        return [str.removeprefix(entry, "kvcache_") for entry in entries 
            if os.path.isfile(os.path.join(self.path, entry)) and entry.startswith("kvcache_")]

    def close(self):
        pass

