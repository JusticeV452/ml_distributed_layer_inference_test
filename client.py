import json
import asyncio
import websockets
import torch
from torch import nn

from common import (
    GET_ARCH, READY, INPUT, SEND_ARCH, message, parse_message, layer_from_dict
)

PARTNER_URL = 'ws://localhost:8000'


async def run_inference():
    async with websockets.connect(PARTNER_URL) as websocket:
        async def get(request_id):
            await websocket.send(message(request_id))
            return parse_message(await websocket.recv())
        
        # get model architecture
        response_dict = await get(GET_ARCH)
        assert response_dict["id"] == SEND_ARCH
        layer = layer_from_dict(response_dict["data"])
        await websocket.send(message(READY))
        
        # Enter inference loop, wait for values to process then reply with result
        while True:
            message_dict = parse_message(await websocket.recv())
            if message_dict["id"] != INPUT:
                continue
            result = layer(message_dict["data"])
            reply = message(SEND_ARCH, result)
            await websocket.send(reply)


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(run_inference())
