import asyncio
import inspect
import os
from typing import List, Optional, Tuple, Union, no_type_check
    
import aiofiles

from lmcache.experimental.memory_management import (MemoryAllocatorInterface,
                                                    MemoryObj)
from lmcache.experimental.protocol import RedisMetadata
from lmcache.experimental.storage_backend.connector.base_connector import \
    RemoteConnector
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey

logger = init_logger(__name__)

# TODO(Jiayi): Use `redis.asyncio`
# NOTE(Jiayi): `redis-py` supports async operations, but data copy
# cannot be avoided. `hiredis` is more lower-level but asyncio is
# not supported.


class FileConnector(RemoteConnector):
    """
    The remote url should start with "file://" and only have one host-port pair
    """

    def __init__(self, path: str, loop: asyncio.AbstractEventLoop,
                 memory_allocator: MemoryAllocatorInterface):
        self.memory_allocator = memory_allocator
        self.path = path
        assert os.path.isdir(self.path)
        os.makedirs(os.path.join(self.path, "metadata"), exist_ok=True)
        os.makedirs(os.path.join(self.path, "object"), exist_ok=True)

    def _key_to_path(self, key: CacheEngineKey) -> (str, str):
        key_str = key.to_string().replace("/", "-")
        return os.path.join(self.path, "metadata", key_str + ".meta"), os.path.join(self.path, "object", key_str + ".pt")

    async def exists(self, key: CacheEngineKey) -> bool:
        metadata_path, _ = self._key_to_path(key)
        ok = os.path.isfile(metadata_path)
        logger.debug(f"FileConnector.exists({key}): {ok}")
        return ok

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        metadata_path, object_path = self._key_to_path(key)
        logger.debug(f"FileConnector.get({key.to_string()}): {metadata_path}")
        if not os.path.isfile(metadata_path):
            return None
        async with aiofiles.open(metadata_path, 'rb') as f:
            metadata_bytes = await f.read()
        metadata = RedisMetadata.deserialize(memoryview(metadata_bytes))

        memory_obj = self.memory_allocator.allocate(
            metadata.shape,
            metadata.dtype,
            metadata.fmt,
        )
        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None

        view = memoryview(memory_obj.byte_array)
        async with aiofiles.open(object_path, 'rb') as f:
            await f.readinto(view)

        return memory_obj

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        metadata_path, object_path = self._key_to_path(key)
        logger.debug(f"FileConnector.put({key.to_string()}): {metadata_path}")

        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        metadata_bytes = RedisMetadata(len(kv_bytes), kv_shape, kv_dtype,
                                             memory_format).serialize()

        async with aiofiles.open(metadata_path, 'wb') as f:
            await f.write(metadata_bytes)
        async with aiofiles.open(object_path, 'wb') as f:
            await f.write(kv_bytes)
        self.memory_allocator.ref_count_down(memory_obj)

    # TODO
    @no_type_check
    async def list(self) -> List[str]:
        pass

    async def close(self):
        pass

