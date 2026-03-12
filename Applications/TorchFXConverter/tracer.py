"""
Torch.FX callback-based tracer for HuggingFace models.

This tracer uses TorchFunctionMode + forward hooks to capture the execution
graph of any HuggingFace model. Leaf modules are preserved as atomic nodes
in the FX graph, which maps directly to NNTrainer layer types.

The LEAF_MODULES tuple defines which PyTorch modules should be treated as
atomic (non-decomposed) nodes. These are chosen to match NNTrainer's
supported layer types.
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

# =============================================================================
# LEAF MODULES for NNTrainer mapping
# =============================================================================
# These modules are preserved as single nodes in the FX graph.
# Each one maps to an NNTrainer layer type.
#
# NNTrainer layer mapping:
#   nn.Linear          -> fully_connected
#   nn.Embedding       -> embedding_layer / tie_word_embeddings
#   nn.LayerNorm       -> layer_norm (BERT/T5)
#   nn.Dropout         -> (skip in inference)
#   nn.ReLU/GELU/SiLU  -> activation (detected from graph)
#
# HuggingFace-specific RMSNorm modules are detected dynamically at runtime
# (since class names vary: LlamaRMSNorm, Qwen2RMSNorm, MistralRMSNorm, etc.)

LEAF_MODULES = (
    # Linear layers -> fully_connected
    nn.Linear,
    # Embedding layers -> embedding_layer
    nn.Embedding,
    # Normalization -> layer_norm
    nn.LayerNorm,
    # Activations (detected as leaf to simplify graph)
    nn.ReLU,
    nn.GELU,
    nn.SiLU,
    nn.Sigmoid,
    nn.Tanh,
    nn.Softmax,
    # Dropout (will be skipped in inference output)
    nn.Dropout,
    # Conv layers (for future models that use convolutions)
    nn.Conv1d,
    nn.Conv2d,
    # Pooling
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.MaxPool1d,
    nn.MaxPool2d,
    # Batch norm
    nn.BatchNorm1d,
    nn.BatchNorm2d,
)


def _is_rmsnorm(module):
    """Check if a module is any variant of RMSNorm.

    HuggingFace models define their own RMSNorm classes with model-specific
    prefixes (LlamaRMSNorm, Qwen2RMSNorm, MistralRMSNorm, etc.).
    We detect them by class name pattern.
    """
    cls_name = type(module).__name__
    return "RMSNorm" in cls_name or "rmsnorm" in cls_name.lower()


def _build_leaf_check(leaf_modules):
    """Build a function that checks if a module should be a leaf node."""
    def is_leaf(module):
        if isinstance(module, leaf_modules):
            return True
        if _is_rmsnorm(module):
            return True
        return False
    return is_leaf


class Tracer(TorchFunctionMode):
    """Callback-based FX graph tracer for HuggingFace models.

    Usage:
        model = AutoModelForCausalLM.from_pretrained("...")
        tracer = Tracer(model)
        with tracer:
            with torch.no_grad():
                model(input_ids)
        tracer.graph.print_tabular()
    """

    def __init__(self, root: nn.Module, leaf_modules=LEAF_MODULES):
        self.root = root
        self.leaf_modules = leaf_modules
        self._is_leaf = _build_leaf_check(leaf_modules)
        self.graph = torch.fx.Graph()
        self.module_stack = []
        self._tensor_to_node = {}
        self.handles = []
        self.tracing_enabled = True
        self._obj_to_path = {}

        # Register all named parameters, buffers, and modules
        for name, obj in itertools.chain(
            root.named_parameters(),
            root.named_buffers(),
            root.named_modules(),
        ):
            self._obj_to_path[id(obj)] = name

        # Store module references for later inspection
        self._modules = dict(root.named_modules())

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
                        raise RuntimeError(f"Unsupported key type: {type(key)}")
                    cur_node = self.graph.call_function(
                        operator.getitem, args=(cur_node, key), kwargs={}
                    )
                    self._tensor_to_node[id(item)] = cur_node

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if not self.tracing_enabled:
            return func(*args, **kwargs)

        # Ignore descriptor methods
        if getattr(func, "__name__", None) == "__get__":
            return func(*args, **kwargs)

        # Filter out __repr__
        if func == Tensor.__repr__:
            return func(*args, **kwargs)

        out = func(*args, **kwargs)

        node_args = tree_map(self._get_node, args)
        node_kwargs = tree_map(self._get_node, kwargs)

        # Generate human-readable name based on module scope
        name_hint = None
        if self.module_stack:
            scope = self.module_stack[-1]
            if scope:
                op_name = func.__name__ if hasattr(func, "__name__") else "op"
                name_hint = f"{scope.replace('.', '_')}_{op_name}"

        # Check if it's a method call
        if len(args) > 0 and getattr(args[0].__class__, func.__name__, None) is func:
            node = self.graph.create_node(
                "call_method", func.__name__,
                args=node_args, kwargs=node_kwargs, name=name_hint
            )
        else:
            node = self.graph.create_node(
                "call_function", func,
                args=node_args, kwargs=node_kwargs, name=name_hint
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
                    if param and param.kind == inspect.Parameter.VAR_POSITIONAL:
                        for i, item in enumerate(arg_value):
                            name_hint = f"arg_{i}"
                            node = self.graph.placeholder(name_hint)
                            node.meta["type"] = type(item)
                            self._register_output(item, node)
                    elif param and param.kind == inspect.Parameter.VAR_KEYWORD:
                        for key, item in arg_value.items():
                            node = self.graph.placeholder(key)
                            node.meta["type"] = type(item)
                            self._register_output(item, node)
                    else:
                        node = self.graph.placeholder(arg_name)
                        node.meta["type"] = type(arg_value)
                        self._register_output(arg_value, node)

            if self._is_leaf(module):
                self.tracing_enabled = False
            else:
                self.module_stack.append(name)

            # Track consumer modules
            for i, arg in itertools.chain(enumerate(args), kwargs.items()):
                for item in pytree.tree_flatten(arg)[0]:
                    node = self._get_node(item)
                    if isinstance(node, torch.fx.Node):
                        node.meta.setdefault("consumer_module", []).append((name, i))

        return hook

    def _post_hook(self, name):
        def hook(module, args, kwargs, output):
            if self._is_leaf(module):
                self.tracing_enabled = True
                node_args = tree_map(self._get_node, args)
                node_kwargs = tree_map(self._get_node, kwargs)

                node = self.graph.call_module(name, args=node_args, kwargs=node_kwargs)
                node.meta["scope"] = name
                node.meta["output_type"] = type(output)
                node.meta["leaf_module"] = True
                node.meta["module_type"] = type(module).__name__
                node.meta["module_class"] = type(module)

                # Store module config for later use by emitter
                if hasattr(module, "weight"):
                    node.meta["has_weight"] = True
                if hasattr(module, "bias") and module.bias is not None:
                    node.meta["has_bias"] = True
                else:
                    node.meta["has_bias"] = False

                # Store module-specific attributes
                if isinstance(module, nn.Linear):
                    node.meta["in_features"] = module.in_features
                    node.meta["out_features"] = module.out_features
                elif isinstance(module, nn.Embedding):
                    node.meta["num_embeddings"] = module.num_embeddings
                    node.meta["embedding_dim"] = module.embedding_dim
                elif isinstance(module, nn.LayerNorm):
                    node.meta["normalized_shape"] = module.normalized_shape
                    node.meta["eps"] = module.eps
                elif _is_rmsnorm(module):
                    node.meta["is_rmsnorm"] = True
                    if hasattr(module, "eps"):
                        node.meta["eps"] = module.eps
                    elif hasattr(module, "variance_epsilon"):
                        node.meta["eps"] = module.variance_epsilon

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
                module.register_forward_pre_hook(self._pre_hook(name), with_kwargs=True)
            )
            self.handles.append(
                module.register_forward_hook(self._post_hook(name), with_kwargs=True)
            )
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()
        return super().__exit__(exc_type, exc_val, exc_tb)

    def get_leaf_modules(self):
        """Return a dict of {name: module} for all leaf modules found during tracing."""
        result = {}
        for node in self.graph.nodes:
            if node.op == "call_module" and node.meta.get("leaf_module"):
                module_name = node.target
                if module_name in self._modules:
                    result[module_name] = self._modules[module_name]
        return result

    def print_graph_summary(self):
        """Print a summary of the traced graph."""
        counts = {}
        for node in self.graph.nodes:
            key = node.op
            if node.op == "call_module" and node.meta.get("module_type"):
                key = f"call_module({node.meta['module_type']})"
            counts[key] = counts.get(key, 0) + 1

        print("\n=== Graph Summary ===")
        print(f"Total nodes: {sum(counts.values())}")
        for key, count in sorted(counts.items()):
            print(f"  {key}: {count}")
        print("=====================\n")
