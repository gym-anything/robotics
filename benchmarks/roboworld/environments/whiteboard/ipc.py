"""Length-delimited, finite JSON. Local renderer IPC never executes received code."""

import json
import struct

LIMIT = 64 * 1024 * 1024


def send(sock, message):
    payload = json.dumps(message, allow_nan=False, separators=(",", ":")).encode()
    if len(payload) > LIMIT:
        raise ValueError("IPC message exceeds 64 MiB")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def receive(sock):
    def read(n):
        data = bytearray()
        while len(data) < n:
            part = sock.recv(n - len(data))
            if not part:
                raise ConnectionError("Renderer connection closed")
            data.extend(part)
        return data

    length = struct.unpack("!I", read(4))[0]
    if length > LIMIT:
        raise ValueError("IPC message exceeds 64 MiB")
    return json.loads(read(length), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
