import re
import ast
import json
import torch
from torch import nn

# constants used in message passing between client and server
GET_ARCH = 0
SEND_ARCH = 1
READY = 2
INPUT = 3


"""
JSONEncoder subclass for serializing 
"""
class StateEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, torch.Tensor):
            return {"TENSOR": o.tolist()}
        return super().default(o)


"""
JSONDecoder subclass for deserializing tensors serialized with StateEncoder
"""
class StateDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(
            self, object_hook=self.object_hook, *args, **kwargs
        )
    
    def object_hook(self, dct: dict):
        if "TENSOR" in dct:
            return torch.Tensor(dct["TENSOR"])
        return dct

"""
Parse message string received in client-server communication,
returns dict with keys of 'id' and 'data'
"""
def parse_message(message: str):
    return json.loads(message, cls=StateDecoder)


"""
Get message string for client-server communication,
JSON serialized dictionary with keys of 'id' and 'data'
"""
def message(message_id: int, data=None):
    return json.dumps({"id": message_id, "data": data}, cls=StateEncoder)


"""
Converts a layer to a dictionary with keys of (type, args, kwargs, state)
where type = name of nn layer, state = layer.state_dict(), and
args and kwargs are initialization parameters for layer
"""
def layer_to_dict(layer: torch.nn.Module):
    # Get class name from layer str represemtation
    layer_name, raw_args = str(layer).split('(', 1)
    
    # split string by ',' not between '(' and ')'
    # https://stackoverflow.com/questions/26633452/how-to-split-by-commas-that-are-not-within-parentheses
    args_strs = re.split(r",\s*(?![^()]*\))", raw_args[:-1])
    
    args = []
    kwargs = {}
    for arg_str in args_strs:
        if '=' in arg_str:
            key, val = arg_str.split('=')
            kwargs[key] = ast.literal_eval(val)
        else:
            args.append(ast.literal_eval(arg_str))
    return {
        "type": layer_name,
        "args": args,
        "kwargs": kwargs,
        "state": layer.state_dict()
    }


"""
Convert a layer dictionary created using `layer_to_dict` back to an nn object
with the same weights as the original layer
"""
def layer_from_dict(layer_dict: dict):
    layer_class = getattr(nn, layer_dict["type"])
    layer = layer_class(*layer_dict["args"], **layer_dict["kwargs"],)
    layer.load_state_dict(layer_dict["state"])
    return layer
