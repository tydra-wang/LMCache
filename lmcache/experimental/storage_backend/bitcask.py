import os
import asyncio
import threading
import aiofiles
import struct

from typing import Optional, Dict

"""
-------------------------------------------
| key_length | value_length | key | value |
-------------------------------------------
"""

class BitCaskReader:
    def __init__(self, path: str, loop: asyncio.AbstractEventLoop):
        print("initing BitCaskReader")
        self.path = path
        assert os.path.isfile(path)
        self.index : Dict[str,int] = {}
        self.index_lock = threading.Lock()
        self.index_task = asyncio.run_coroutine_threadsafe(self._maintain_index(), loop)
        # self.index_task = loop.create_task(self._maintain_index())

    async def _read(self, reader, n: int) -> bytes:
        buffer = bytearray()
        while len(buffer) < n:
            chunk = await reader.read(n - len(buffer))
            if not chunk:
                await asyncio.sleep(1)
            buffer.extend(chunk)
        assert len(buffer) == n
        return bytes(buffer)

    async def _maintain_index(self):
        print("start updating index ...")
        async with aiofiles.open(self.path, 'rb') as file:
            while True:
                pos = await file.tell()
                key_length_bytes = await self._read(file, 4)
                key_length = struct.unpack('I', key_length_bytes)[0]
                assert key_length > 0 and key_length < 1024
                key = (await self._read(file, key_length)).decode()
                value_length = struct.unpack('I', (await self._read(file, 4)))[0]
                await file.seek(value_length, os.SEEK_CUR)
                print(f"Add index: {key} -> {pos+4+key_length}")
                with self.index_lock:
                    self.index[key] = pos + 4 + key_length

    def get(self, key: str) -> Optional[bytes]:
        with self.index_lock:
            if key not in self.index:
                return None
            pos = self.index[key]
        with open(self.path, 'rb') as file:
            file.seek(pos)
            value_length = struct.unpack('I', file.read(4))[0]
            value = file.read(value_length)
        return value

    def contains(self, key: str) -> bool:
        with self.index_lock:
            return key in self.index
    
    def close(self):
        self.index_task.cancel()

class BitCaskWriter:
    def __init__(self, path: str):
        self.file = open(path, 'ab')

    def put(self, key: str, value: bytes):
        key_encoded = key.encode()
        self.file.write(struct.pack('I', len(key_encoded)))
        self.file.write(key_encoded)
        self.file.write(struct.pack('I', len(value)))
        self.file.write(value)
        self.file.flush()

    def close(self):
        self.file.close()

