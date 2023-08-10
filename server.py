import json
import asyncio
import torch
from torch import nn
from websockets.server import serve

from common import (
    GET_ARCH, READY, INPUT, SEND_ARCH, message, parse_message, layer_to_dict
)

PORT = 8000
BATCH_SIZE = 2
INP_SIZE = 5
HIDDEN_SIZE = 10
OUTPUT_SIZE = 1

layer1 = nn.Sequential(
    nn.Linear(INP_SIZE, HIDDEN_SIZE),
    nn.ReLU()
)
layer2 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
model = nn.Sequential(layer1, layer2)

async def handler(websocket):
    async for recv_message in websocket:
        message_dict = parse_message(recv_message)
        message_type = message_dict["id"]
        
        if message_type == GET_ARCH:
            # Send layer arcitechture to client
            await websocket.send(message(SEND_ARCH, layer_to_dict(layer2)))
        
        while message_type == READY:
            inp = torch.rand(BATCH_SIZE, INP_SIZE)
            layer_out = layer1(inp)
            print("Getting second layer result from client...")
            await websocket.send(message(INPUT, layer_out))
            result = parse_message(await websocket.recv())["data"]
            print("Got:", result, "Expected:", model(inp))

async def main():
    async with serve(handler, "localhost", PORT):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
