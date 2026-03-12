#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Runtime-based FX graph tracer using TorchFunctionMode.

Unlike symbolic tracing, this tracer runs the model with real inputs
and records the actual execution path, handling conditionals and
dynamic control flow correctly.
"""

import inspect
import itertools
import operator

import torch
import torch.fx
import torch.utils._pytree as pytree
from torch import Tensor, nn
from torch.overrides import TorchFunctionMode
from torch.utils._pytree import tree_map

# Default leaf modules - these are treated as atomic nodes in the graph.
# When a module is a leaf, its internal ops are NOT traced; only its
# forward() call appears as a single call_module node.
LEAF_MODULES = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.ReLU,
    nn.ReLU6,
    nn.GELU,
    nn.SiLU,
    nn.Hardswish,
    nn.PReLU,
    nn.LeakyReLU,
    nn.Sigmoid,
    nn.Tanh,
    nn.Embedding,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool1d,
    nn.MaxPool2d,
    nn.MaxPool1d,
    nn.Dropout,
    nn.Dropout2d,
    nn.Dropout3d,
)


class Tracer(TorchFunctionMode):
    """Runtime-based FX graph tracer.

    Uses TorchFunctionMode to intercept all torch function calls during
    actual model execution with real inputs. Module-level hooks capture
    call_module nodes for leaf modules.

    Usage:
        tracer = Tracer(model, leaf_modules=LEAF_MODULES)
        with tracer:
            output = model(input_ids)
        graph = tracer.graph
        graph.print_tabular()
    """

    def __init__(self, root: nn.Module, leaf_modules=LEAF_MODULES):
        self.root = root
        self.leaf_modules = leaf_modules
        self.graph = torch.fx.Graph()
        self.module_stack = []
        self._tensor_to_node = {}
        self.handles = []
        self.tracing_enabled = True
        self._obj_to_path = {}

        for name, obj in list(root.named_parameters()) + list(
            root.named_buffers()
        ) + list(root.named_modules()):
            self._obj_to_path[id(obj)] = name

    def _get_node(self, obj):
        if isinstance(obj, Tensor):
            if id(obj) in self._tensor_to_node:
                return self._tensor_to_node[id(obj)]

            if id(obj) in self._obj_to_path:
                name = self._obj_to_path[id(obj)]
                node = self.graph.get_attr(name)
                self._tensor_to_node[id(obj)] = node
                return node

            name = f"_tensor_constant_{len(self._obj_to_path)}"
            self.root.register_buffer(name, obj.detach().clone())
            self._obj_to_path[id(obj)] = name
            node = self.graph.get_attr(name)
            self._tensor_to_node[id(obj)] = node
            return node
        return obj

    def _register_output(self, out, node):
        for path, item in pytree.tree_flatten_with_path(out)[0]:
            if isinstance(item, Tensor):
                cur_node = node
                for key in path:
                    if isinstance(key, pytree.SequenceKey):
                        key = key.idx
                    elif isinstance(key, pytree.MappingKey):
                        key = key.key
                    elif isinstance(key, pytree.GetAttrKey):
                        key = key.name
                    else:
                        raise RuntimeError(
                            f"Unsupported key type: {type(key)}"
                        )
                    cur_node = self.graph.call_function(
                        operator.getitem, args=(cur_node, key), kwargs={}
                    )
                    self._tensor_to_node[id(item)] = cur_node

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if not self.tracing_enabled:
            return func(*args, **kwargs)

        # Ignore descriptor methods like __get__
        if getattr(func, "__name__", None) == "__get__":
            return func(*args, **kwargs)

        # Filter out __repr__
        if func == Tensor.__repr__:
            return func(*args, **kwargs)

        out = func(*args, **kwargs)

        node_args = tree_map(self._get_node, args)
        node_kwargs = tree_map(self._get_node, kwargs)

        # Generate a human-readable name based on the module scope
        name_hint = None
        if self.module_stack:
            scope = self.module_stack[-1]
            if scope:
                op_name = (
                    func.__name__ if hasattr(func, "__name__") else "op"
                )
                name_hint = f"{scope.replace('.', '_')}_{op_name}"

        # Check if it's a method call
        if len(args) > 0 and getattr(
            args[0].__class__, func.__name__, None
        ) is func:
            node = self.graph.create_node(
                "call_method",
                func.__name__,
                args=node_args,
                kwargs=node_kwargs,
                name=name_hint,
            )
        else:
            node = self.graph.create_node(
                "call_function",
                func,
                args=node_args,
                kwargs=node_kwargs,
                name=name_hint,
            )

        if self.module_stack:
            node.meta["scope"] = self.module_stack[-1]
            node.meta["output_type"] = type(out)

        self._register_output(out, node)

        return out

    def _pre_hook(self, name):
        def hook(module, args, kwargs):
            if name == "":  # root module
                sig = inspect.signature(module.forward)
                bound = sig.bind(*args, **kwargs)

                for arg_name, arg_value in bound.arguments.items():
                    param = sig.parameters.get(arg_name)
                    if (
                        param
                        and param.kind == inspect.Parameter.VAR_POSITIONAL
                    ):
                        for i, item in enumerate(arg_value):
                            name_hint = f"arg_{i}"
                            node = self.graph.placeholder(name_hint)
                            node.meta["type"] = type(item)
                            self._register_output(item, node)
                    elif (
                        param
                        and param.kind == inspect.Parameter.VAR_KEYWORD
                    ):
                        for key, item in arg_value.items():
                            node = self.graph.placeholder(key)
                            node.meta["type"] = type(item)
                            self._register_output(item, node)
                    else:
                        node = self.graph.placeholder(arg_name)
                        node.meta["type"] = type(arg_value)
                        self._register_output(arg_value, node)

            if isinstance(module, self.leaf_modules):
                self.tracing_enabled = False
            else:
                self.module_stack.append(name)

            for i, arg in itertools.chain(
                enumerate(args), kwargs.items()
            ):
                for item in pytree.tree_flatten(arg)[0]:
                    node = self._get_node(item)
                    if isinstance(node, torch.fx.Node):
                        node.meta.setdefault("consumer_module", []).append(
                            (name, i)
                        )

        return hook

    def _post_hook(self, name):
        def hook(module, args, kwargs, output):
            if isinstance(module, self.leaf_modules):
                self.tracing_enabled = True
                node_args = tree_map(self._get_node, args)
                node_kwargs = tree_map(self._get_node, kwargs)

                node = self.graph.call_module(
                    name, args=node_args, kwargs=node_kwargs
                )
                node.meta["scope"] = name
                node.meta["output_type"] = type(output)
                node.meta["leaf_module"] = True

                self._register_output(output, node)
            elif self.module_stack:
                self.module_stack.pop()

            if isinstance(output, Tensor):
                cur_node = self._get_node(output)
                if cur_node.meta.get("scope", None) == name:
                    cur_node.meta["producer_module"] = name
            elif isinstance(output, (tuple, list)):
                for key, item in enumerate(output):
                    if isinstance(item, Tensor):
                        cur_node = self._get_node(item)
                        if cur_node.meta.get("scope", None) == name:
                            cur_node.meta["producer_module"] = (name, key)

            if name == "":
                output_args = tree_map(self._get_node, output)
                self.graph.output(output_args)

        return hook

    def __enter__(self):
        for name, module in self.root.named_modules():
            self.handles.append(
                module.register_forward_pre_hook(
                    self._pre_hook(name), with_kwargs=True
                )
            )
            self.handles.append(
                module.register_forward_hook(
                    self._post_hook(name), with_kwargs=True
                )
            )
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()
        return super().__exit__(exc_type, exc_val, exc_tb)
