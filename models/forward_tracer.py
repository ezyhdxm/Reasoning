import torch
import torch.nn as nn
from typing import Dict, Optional
from dataclasses import dataclass

# adapted from https://github.com/roeehendel/icl_task_vectors/

def get_nested_attr(obj, attr_path):
    attrs = attr_path.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj

def find_longest_modulelist(model, path=""):
    """
    Recursively find the longest nn.ModuleList in a PyTorch model.
    Args:
        model: PyTorch model.
        path: Current path in the model (used for recursion).
    Returns:
        Tuple with path and length of the longest nn.ModuleList found.
    """
    longest_path, longest_len = path, 0

    for name, child in model.named_children():
        if isinstance(child, nn.ModuleList) and len(child) > longest_len:
            longest_len = len(child)
            longest_path = f"{path}.{name}" if path else name

        # Recursively check the child's children
        child_path, child_len = find_longest_modulelist(child, f"{path}.{name}" if path else name)
        if child_len > longest_len:
            longest_len, longest_path = child_len, child_path

    return longest_path, longest_len

def find_module(block, keywords):
    """
    Try to find a module in a transformer block.
    Args:
        block: Transformer block (nn.Module).
        keywords: List of possible module names (str).
    Returns:
        The found module if found, else None.
    """
    for name, module in block.named_modules():
        if any(keyword in name for keyword in keywords):
            return module
    submodule_names = [name for name, _ in block.named_modules()]
    raise ValueError(f"Could not find keywords {keywords} in: {submodule_names}")

def get_embedding_layer(model):
    # model_type = model.__class__.__name__

    keywords = ["emb", "wte"]
    return find_module(model, keywords)

def get_layers_path(model):
    longest_path, longest_len = find_longest_modulelist(model)
    return longest_path

def get_layers(model):
    # model_type = model.__class__.__name__

    longest_path = get_layers_path(model)
    return get_nested_attr(model, longest_path)

def get_attention_layers(model):
    # model_type = model.__class__.__name__

    layers = get_layers(model)
    keywords = ["attention", "attn", "MHA"]
    attention_layers = [find_module(layer, keywords) for layer in layers]
    return attention_layers

def get_mlp_layers(model):
    # model_type = model.__class__.__name__

    layers = get_layers(model)
    mlp_keywords = ["mlp", "feedforward", "ffn"]
    mlp_layers = [find_module(layer, mlp_keywords) for layer in layers]
    return mlp_layers

@dataclass
class ResidualStream:
    hidden: torch.Tensor
    attn: torch.Tensor
    mlp: torch.Tensor


class ForwardTrace:
    def __init__(self):
        self.residual_stream: Optional[ResidualStream] = ResidualStream(hidden=[], attn=[], mlp=[])
        self.attentions: Optional[torch.Tensor] = None

class ForwardTracer:
    def __init__(self, model, forward_trace: ForwardTrace, with_submodules: bool = False):
        self._model = model
        self._forward_trace = forward_trace
        self._with_submodules = with_submodules

        self._layers = get_layers(model)
        self._attn_layers = get_attention_layers(model)
        self._mlp_layers = get_mlp_layers(model)

        self._hooks = []
    
    def __enter__(self): 
        self._register_forward_hooks()
    
    def __exit__(self, exc_type, exc_value, traceback):
        for hook in self._hooks:
            hook.remove()
        
        if exc_type is not None:
            residual_stream = self._forward_trace.residual_stream

            if residual_stream.hidden[0] == []:
                residual_stream.hidden.pop(0)
            
            for key in residual_stream.__dataclass_fields__.keys():
                acts = getattr(residual_stream, key)
    

    def _register_forward_hooks(self): # obtain activations and attention maps
        model = self._model
        hooks = self._hooks

        residual_stream = self._forward_trace.residual_stream

        def store_activations(residual_stream, acts_type, layer_num):
            def hook(model, inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                out = out.float().to("cpu", non_blocking=True)

                acts = getattr(residual_stream, acts_type)
                while len(acts) <= layer_num:
                    acts.append([])
                try: 
                    acts[layer_num].append(out)
                except IndexError:
                    print(len(acts), layer_num)

            return hook
        
        def store_attentions(layer_num):
            def hook(model, inp, out):
                attention_maps = out[1]
                print(attention_maps)
                attention_maps = attention_maps.to("cpu", non_blocking=True).float()
                print(attention_maps)
                self._forward_trace.attentions[layer_num] = attention_maps

            return hook
        
        embedding_hook = get_embedding_layer(self._model).register_forward_hook(
            store_activations(residual_stream, "hidden", 0)
        )
        hooks.append(embedding_hook)

        for i, layer in enumerate(self._layers):
            hidden_states_hook = layer.register_forward_hook(store_activations(residual_stream, "hidden", i + 1))
            hooks.append(hidden_states_hook)

        if self._with_submodules:
            for i, mlp_layer in enumerate(self._mlp_layers):
                mlp_res_hook = mlp_layer.register_forward_hook(store_activations(residual_stream, "mlp", i))
                hooks.append(mlp_res_hook)

            for i, attn_layer in enumerate(self._attn_layers):
                attn_res_hook = attn_layer.register_forward_hook(store_activations(residual_stream, "attn", i))
                hooks.append(attn_res_hook)
                attn_attentions_hook = attn_layer.register_forward_hook(store_attentions(i))
                hooks.append(attn_attentions_hook)