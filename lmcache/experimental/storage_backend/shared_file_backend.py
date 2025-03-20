import asyncio
import aiohttp
import threading
from concurrent.futures import Future
from typing import List, Optional

from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.experimental.lookup_server import LookupServerInterface
from lmcache.experimental.memory_management import (MemoryAllocatorInterface,
                                                    MemoryObj)
from lmcache.experimental.storage_backend.abstract_backend import \
    StorageBackendInterface
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey

from lmcache.experimental.storage_backend.bitcask import BitCaskReader
from lmcache.experimental.protocol import SharedFileMetadata, SHARED_FILE_METADATA_LEN

logger = init_logger(__name__)


class Metadata:
    offset: int


class SharedFileBackend(StorageBackendInterface):

    def __init__(
        self,
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        memory_allocator: MemoryAllocatorInterface,
        dst_device: str = "cuda",
        lookup_server: Optional[LookupServerInterface] = None,
    ):
        assert config.shared_file is not None
        self.bitcask_reader = BitCaskReader(config.shared_file, loop)
        self.wserver = config.shared_file_wserver

        self.loop = loop
        self.memory_allocator = memory_allocator

        self.put_tasks_lock = threading.Lock()
        self.put_tasks: List[CacheEngineKey] = []

    def __str__(self):
        return self.__class__.__name__

    def contains(self, key: CacheEngineKey) -> bool:
        """
        Check whether key is in the storage backend. 
        """
        return self.bitcask_reader.contains(key.to_string())

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """
        Check whether key is in the ongoing put tasks. 
        """
        with self.put_tasks_lock:
            return key in self.put_tasks

    def submit_put_task(self, key: CacheEngineKey,
                        obj: MemoryObj) -> Optional[Future]:
        """
        An async function to put the MemoryObj into the storage backend.

        :param CacheEngineKey key: The key of the MemoryObj.
        :param MemoryObj obj: The MemoryObj to be stored.
        
        :return: a future object
        """
        assert obj.tensor is not None
        self.memory_allocator.ref_count_up(obj)

        with self.put_tasks_lock:
            self.put_tasks.append(key)

        future = asyncio.run_coroutine_threadsafe(self.async_save_bytes(key, obj), self.loop)

        lambda_callback = lambda f: \
                self.put_callback(f, key)
        future.add_done_callback(lambda_callback)
        return future

    def put_callback(self, future: Future, key: CacheEngineKey):
        """
        Callback function for put tasks.
        """
        with self.put_tasks_lock:
            self.put_tasks.remove(key)

    async def async_save_bytes(self, key: CacheEngineKey, obj: MemoryObj):
        url = f"{self.wserver}/{self._key_to_bitcask_key(key)}"
        logger.debug(f"sending to {url}")

        kv_bytes = obj.byte_array
        kv_shape = obj.get_shape()
        kv_dtype = obj.get_dtype()
        memory_format = obj.get_memory_format()
        metadata_bytes = SharedFileMetadata(len(kv_bytes), kv_shape, kv_dtype, memory_format).serialize()
        body = b''.join([metadata_bytes, kv_bytes])

        async with aiohttp.ClientSession() as session:
            # reader = aiohttp.StreamReader()
            # await reader.feed_data(metadata_bytes)
            # await reader.feed_data(kv_bytes)
            # reader.feed_eof()
            async with session.post(url, data=body) as response:
                if response.status != 200:
                    logger.debug(f"Request failed with status code {response.status}")

        self.memory_allocator.ref_count_down(obj)

    def submit_prefetch_task(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        """
        An async function to get the MemoryObj from the storage backend.

        :param CacheEngineKey key: The key of the MemoryObj.

        :return: a future object. None if the key does not exist.
        """
        pass

    def _key_to_bitcask_key(
        self,
        key: CacheEngineKey,
    ) -> str:
        return key.to_string().replace("/", "-")

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """
        A blcocking function to get the kv cache from the storage backend.
        
        :param CacheEngineKey key: The key of the MemoryObj.
        
        :return: MemoryObj. None if the key does not exist.
        """
        value = self.bitcask_reader.get(self._key_to_bitcask_key(key))
        if not value:
            return None
        metadata = SharedFileMetadata.deserialize(value[:SHARED_FILE_METADATA_LEN])

        assert metadata.length + SHARED_FILE_METADATA_LEN == len(value)

        memory_obj = self.memory_allocator.allocate(
            metadata.shape,
            metadata.dtype,
            metadata.fmt,
        )
        if memory_obj is None:
            logger.warning("Failed to allocate memory")
            return None
        
        view = memoryview(memory_obj.byte_array)
        view[:metadata.length] = value[SHARED_FILE_METADATA_LEN:]
        return memory_obj 
        

    def close(self, ) -> None:
        """
        Close the storage backend.
        """
        self.bitcask_reader.close()
