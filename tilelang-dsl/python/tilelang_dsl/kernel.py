"""Kernel descriptor surface for TileLang DSL v1."""

from __future__ import annotations

import inspect
import textwrap
import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .types import (
    MemorySpace,
    ScalarType,
    TensorView,
    Tile,
    TileConfig,
    TileSpecialization,
    TypeVariable,
    WildcardType,
)
from .frontend_ast import build_frontend_kernel_node
from .lowering import lower_semantic_kernel
from .semantic import analyze_frontend_kernel


_UNSET = object()
_MATCHER_FOLLOW_UP_CHANGE = "extend-tilelang-dsl-matcher-and-advanced-surface"
_V1_ALLOWED_TOPLEVEL_PTO_CALLS = {
    "strict_vecscope",
    "dma_load",
    "dma_store",
    "set_flag",
    "wait_flag",
    "pipe_barrier",
    "barrier",
}
_V1_ALLOWED_VECSCOPE_PTO_CALLS = {
    "make_mask",
    "vlds",
    "vsts",
    "vabs",
    "vrelu",
    "vexp",
    "vnot",
    "vadd",
    "vsub",
    "vmul",
    "vdiv",
    "vmax",
    "vmin",
    "vand",
    "vor",
    "vxor",
    "vadds",
    "vsubs",
    "vmuls",
    "vdivs",
    "vmaxs",
    "vmins",
}


def _unsupported_feature_message(feature: str) -> str:
    return (
        f"{feature} is not supported in TileLang DSL v1; "
        f"see follow-up change `{_MATCHER_FOLLOW_UP_CHANGE}`"
    )


def _reject_unsupported_decorator_feature(name: str, value: Any) -> None:
    if value is _UNSET:
        return
    raise ValueError(_unsupported_feature_message(f"decorator feature `{name}`"))


def _reject_unsupported_dtype_feature(dtype: Any) -> None:
    if isinstance(dtype, WildcardType):
        raise ValueError(
            _unsupported_feature_message(f"dtype wildcard `{dtype.name}`")
        )
    if isinstance(dtype, TypeVariable):
        raise ValueError(
            _unsupported_feature_message(f"dtype type variable `{dtype.name}`")
        )


class TileLangFrontendError(ValueError):
    """Source-located frontend diagnostic for TileLang DSL."""

    def __init__(self, path: str, line: int, column: int, message: str):
        self.path = path
        self.line = line
        self.column = column
        self.message = message
        super().__init__(f"{path}:{line}:{column}: {message}")


@dataclass(frozen=True)
class _FunctionSourceInfo:
    path: str
    start_line: int
    function_def: ast.FunctionDef

    def location(self, node: ast.AST) -> tuple[int, int]:
        line = self.start_line + getattr(node, "lineno", 1) - 1
        column = getattr(node, "col_offset", 0) + 1
        return line, column

    def error(self, node: ast.AST, message: str) -> TileLangFrontendError:
        line, column = self.location(node)
        return TileLangFrontendError(self.path, line, column, message)

    def parameter_node(self, param_name: str) -> ast.AST | None:
        for arg in self.function_def.args.args:
            if arg.arg == param_name:
                return arg.annotation or arg
        return None


class _KernelBodyValidator(ast.NodeVisitor):
    def __init__(self, source_info: _FunctionSourceInfo):
        self.source_info = source_info
        self._vecscope_depth = 0

    def validate(self) -> None:
        for stmt in self.source_info.function_def.body:
            self.visit(stmt)

    def visit_While(self, node: ast.While) -> None:
        raise self.source_info.error(node, "unsupported Python syntax `while` in TileLang DSL v1")

    def visit_ListComp(self, node: ast.ListComp) -> None:
        raise self.source_info.error(
            node, "unsupported Python syntax `list comprehension` in TileLang DSL v1"
        )

    def visit_DictComp(self, node: ast.DictComp) -> None:
        raise self.source_info.error(
            node, "unsupported Python syntax `dict comprehension` in TileLang DSL v1"
        )

    def visit_SetComp(self, node: ast.SetComp) -> None:
        raise self.source_info.error(
            node, "unsupported Python syntax `set comprehension` in TileLang DSL v1"
        )

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        raise self.source_info.error(
            node, "unsupported Python syntax `generator expression` in TileLang DSL v1"
        )

    def visit_For(self, node: ast.For) -> None:
        if not isinstance(node.target, ast.Name):
            raise self.source_info.error(node.target, "for target must be a single name")
        if not isinstance(node.iter, ast.Call) or not isinstance(node.iter.func, ast.Name):
            raise self.source_info.error(node.iter, "only Python range(lb, ub, step) loops are supported")
        if node.iter.func.id != "range":
            raise self.source_info.error(node.iter, "only Python range(lb, ub, step) loops are supported")
        if len(node.iter.args) != 3:
            raise self.source_info.error(node.iter, "range() expects exactly 3 arguments in TileLang DSL v1")
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_If(self, node: ast.If) -> None:
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_With(self, node: ast.With) -> None:
        if len(node.items) != 1:
            raise self.source_info.error(node, "only single with item is supported in TileLang DSL v1")
        item = node.items[0]
        if not isinstance(item.context_expr, ast.Call):
            raise self.source_info.error(item.context_expr, "with context must be a call in TileLang DSL v1")
        if not (
            isinstance(item.context_expr.func, ast.Attribute)
            and isinstance(item.context_expr.func.value, ast.Name)
            and item.context_expr.func.value.id == "pto"
            and item.context_expr.func.attr == "strict_vecscope"
        ):
            raise self.source_info.error(
                item.context_expr,
                "only pto.strict_vecscope is supported as a with-context in TileLang DSL v1",
            )
        if not isinstance(item.optional_vars, ast.Tuple):
            raise self.source_info.error(item, "pto.strict_vecscope requires tuple binding in 'as'")
        for elt in item.optional_vars.elts:
            if not isinstance(elt, ast.Name):
                raise self.source_info.error(elt, "pto.strict_vecscope bindings must be names")
        self._vecscope_depth += 1
        try:
            for stmt in node.body:
                self.visit(stmt)
        finally:
            self._vecscope_depth -= 1

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == "pto" and node.func.attr in _V1_ALLOWED_TOPLEVEL_PTO_CALLS:
                return
            if node.func.value.id == "pto" and node.func.attr in _V1_ALLOWED_VECSCOPE_PTO_CALLS:
                if self._vecscope_depth <= 0:
                    raise self.source_info.error(
                        node,
                        f"vector op surface `pto.{node.func.attr}` requires explicit pto.strict_vecscope in TileLang DSL v1",
                    )
                return
            if node.func.value.id == "pto":
                raise self.source_info.error(
                    node,
                    f"unsupported op surface `pto.{node.func.attr}` in TileLang DSL v1",
                )
            raise self.source_info.error(
                node,
                f"arbitrary external call `{node.func.value.id}.{node.func.attr}` is not supported "
                "in TileLang DSL v1",
            )

        if isinstance(node.func, ast.Name):
            if node.func.id == "range":
                return
            raise self.source_info.error(
                node,
                f"arbitrary external call `{node.func.id}` is not supported in TileLang DSL v1",
            )

        raise self.source_info.error(
            node,
            "unsupported call surface in TileLang DSL v1",
        )


def _load_function_source_info(py_fn: Callable[..., Any]) -> _FunctionSourceInfo | None:
    try:
        source_lines, start_line = inspect.getsourcelines(py_fn)
        path = inspect.getsourcefile(py_fn) or inspect.getfile(py_fn)
    except (OSError, IOError, TypeError):
        return None

    source = textwrap.dedent("".join(source_lines))
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == py_fn.__name__:
            return _FunctionSourceInfo(path=path, start_line=start_line, function_def=node)
    return None


def _validate_function_body(source_info: _FunctionSourceInfo | None) -> None:
    if source_info is None:
        return
    _KernelBodyValidator(source_info).validate()


def _raise_tile_param_error(
    source_info: _FunctionSourceInfo | None,
    param_name: str,
    message: str,
    fallback_exception: type[Exception] = ValueError,
) -> None:
    if source_info is not None:
        node = source_info.parameter_node(param_name)
        if node is not None:
            raise source_info.error(node, message)
    raise fallback_exception(message)


def _freeze_dtypes(dtypes: Any) -> tuple[tuple[Any, ...], ...]:
    if not isinstance(dtypes, (list, tuple)):
        raise TypeError("dtypes must be a sequence of signature tuples")

    frozen_signatures = []
    for signature in dtypes:
        if not isinstance(signature, (list, tuple)):
            raise TypeError("each dtypes entry must be a signature tuple")
        frozen_signature = tuple(signature)
        for dtype in frozen_signature:
            _reject_unsupported_dtype_feature(dtype)
        frozen_signatures.append(frozen_signature)

    if not frozen_signatures:
        raise ValueError("dtypes must contain at least one signature tuple")

    if len(frozen_signatures) != 1:
        raise ValueError(
            _unsupported_feature_message("multiple dtypes signatures")
        )

    return tuple(frozen_signatures)


@dataclass(frozen=True)
class BoundKernelParameter:
    """One parameter after v1 monomorphic dtype binding."""

    name: str
    kind: str
    annotation: Any
    dtype: ScalarType

    @property
    def element_dtype(self) -> ScalarType | None:
        if self.kind in ("tensorview", "tile"):
            return self.dtype
        return None


@dataclass(frozen=True)
class VKernelDescriptor:
    """Descriptor returned by `@tilelang_dsl.vkernel`."""

    target: str
    op: str
    dtypes: tuple[tuple[Any, ...], ...]
    name: str
    verify_enabled: bool
    parameters: tuple[BoundKernelParameter, ...]
    _py_fn: Callable[..., Any] = field(repr=False)
    _source_info: _FunctionSourceInfo | None = field(repr=False, compare=False, default=None)
    specializations: tuple[tuple[str, TileSpecialization], ...] = ()

    @property
    def py_fn(self) -> Callable[..., Any]:
        return self._py_fn

    @property
    def dtype_signature(self) -> tuple[ScalarType, ...]:
        return self.dtypes[0]

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "op": self.op,
            "dtypes": self.dtypes,
            "name": self.name,
            "verify": self.verify_enabled,
        }

    @property
    def tile_parameters(self) -> tuple[BoundKernelParameter, ...]:
        return tuple(param for param in self.parameters if param.kind == "tile")

    @property
    def specializations_by_name(self) -> dict[str, TileSpecialization]:
        return dict(self.specializations)

    def specialize(self, **bindings: Any) -> "VKernelDescriptor":
        tile_params = {param.name: param for param in self.tile_parameters}
        if not tile_params:
            if bindings:
                unknown = ", ".join(sorted(bindings))
                raise TypeError(
                    f"specialize() received bindings for non-Tile parameters: {unknown}"
                )
            return self

        unknown = sorted(set(bindings) - set(tile_params))
        if unknown:
            unknown_names = ", ".join(unknown)
            raise TypeError(
                f"specialize() only accepts bare Tile parameters; got: {unknown_names}"
            )

        updated = self.specializations_by_name
        for name, binding in bindings.items():
            updated[name] = _coerce_tile_specialization(name, binding, self._source_info)

        return VKernelDescriptor(
            target=self.target,
            op=self.op,
            dtypes=self.dtypes,
            name=self.name,
            verify_enabled=self.verify_enabled,
            parameters=self.parameters,
            _source_info=self._source_info,
            specializations=tuple(sorted(updated.items())),
            _py_fn=self._py_fn,
        )

    def _require_specialized_tiles(self, api_name: str) -> None:
        tile_names = [param.name for param in self.tile_parameters]
        if not tile_names:
            return

        specialized = self.specializations_by_name
        missing = [name for name in tile_names if name not in specialized]
        if missing:
            missing_names = ", ".join(missing)
            _raise_tile_param_error(
                self._source_info,
                missing[0],
                f"{api_name}() requires specialize() bindings for bare Tile parameters: "
                f"{missing_names}",
            )

    def _build_authoring_module(self):
        frontend_kernel = build_frontend_kernel_node(self)
        semantic_kernel = analyze_frontend_kernel(frontend_kernel)
        return lower_semantic_kernel(semantic_kernel)

    def mlir_text(self) -> str:
        self._require_specialized_tiles("mlir_text")
        return self._build_authoring_module().render()

    def mlir_module(self) -> "MaterializedMLIRModule":
        self._require_specialized_tiles("mlir_module")
        return MaterializedMLIRModule(self.mlir_text())

    def verify(self) -> bool:
        self._require_specialized_tiles("verify")
        self.mlir_module()
        return True

    def emit(self, path: str | Path) -> None:
        self._require_specialized_tiles("emit")
        output_path = Path(path)
        output_path.write_text(self.mlir_text(), encoding="utf-8")


@dataclass(frozen=True)
class MaterializedMLIRModule:
    text: str

    def __str__(self) -> str:
        return self.text

    def verify(self) -> bool:
        return True


def _validate_target(target: str) -> str:
    if not isinstance(target, str):
        raise TypeError("target must be a string")
    if target != "a5":
        raise ValueError("TileLang DSL v1 currently only supports target='a5'")
    return target


def _validate_op(op: Any) -> str:
    if not isinstance(op, str) or not op:
        raise TypeError("op must be a non-empty string")
    return op


def _validate_name(py_fn: Callable[..., Any], name: Any) -> str:
    if name is None:
        return py_fn.__name__
    if not isinstance(name, str) or not name:
        raise TypeError("name must be a non-empty string")
    return name


def _validate_verify(verify: Any) -> bool:
    if not isinstance(verify, bool):
        raise TypeError("verify must be a bool")
    return verify


def _coerce_memory_space(value: Any, param_name: str) -> MemorySpace:
    if isinstance(value, MemorySpace):
        return value
    if isinstance(value, str):
        normalized = value.strip().upper()
        try:
            return MemorySpace[normalized]
        except KeyError as exc:
            raise ValueError(
                f"specialization for '{param_name}' uses unsupported memory_space {value!r}"
            ) from exc
    raise TypeError(
        f"specialization for '{param_name}' must provide MemorySpace or string memory_space"
    )


def _coerce_tile_config(value: Any, param_name: str) -> TileConfig | None:
    if value is None:
        return None
    if isinstance(value, TileConfig):
        return value
    if isinstance(value, dict):
        return TileConfig.from_mapping(value)
    raise TypeError(
        f"specialization for '{param_name}' must provide TileConfig, dict, or None for config"
    )


def _coerce_tile_specialization(
    param_name: str,
    binding: Any,
    source_info: _FunctionSourceInfo | None,
) -> TileSpecialization:
    if isinstance(binding, TileSpecialization):
        spec = binding
    elif isinstance(binding, dict):
        if "shape" not in binding:
            _raise_tile_param_error(
                source_info,
                param_name,
                f"specialization for '{param_name}' must provide a static physical Tile shape",
                TypeError,
            )
        if "memory_space" not in binding:
            _raise_tile_param_error(
                source_info,
                param_name,
                f"specialization for '{param_name}' must provide memory_space",
                TypeError,
            )
        spec = TileSpecialization(
            shape=tuple(binding["shape"]),
            memory_space=_coerce_memory_space(binding["memory_space"], param_name),
            config=_coerce_tile_config(binding.get("config"), param_name),
        )
    else:
        _raise_tile_param_error(
            source_info,
            param_name,
            f"specialization for '{param_name}' must be a TileSpecialization or dict",
            TypeError,
        )

    if not spec.shape:
        _raise_tile_param_error(
            source_info,
            param_name,
            f"illegal Tile profile for '{param_name}': shape must be non-empty",
        )
    for dim in spec.shape:
        if not isinstance(dim, int) or isinstance(dim, bool):
            _raise_tile_param_error(
                source_info,
                param_name,
                f"dynamic physical Tile shape is not supported for '{param_name}'",
                TypeError,
            )
        if dim <= 0:
            _raise_tile_param_error(
                source_info,
                param_name,
                f"illegal Tile profile for '{param_name}': dimensions must be positive",
            )
    if len(spec.shape) not in (1, 2):
        _raise_tile_param_error(
            source_info,
            param_name,
            f"illegal Tile profile for '{param_name}': v1 only supports rank-1 or rank-2 Tile shapes",
        )
    if spec.memory_space != MemorySpace.UB:
        _raise_tile_param_error(
            source_info,
            param_name,
            f"illegal Tile profile for '{param_name}': v1 only supports MemorySpace.UB",
        )
    return spec


def _validate_scalar_dtype(dtype: Any, param_name: str) -> ScalarType:
    if not isinstance(dtype, ScalarType):
        raise TypeError(
            f"dtypes entry for parameter '{param_name}' must be a TileLang scalar dtype"
        )
    return dtype


def _bind_parameter(
    param: inspect.Parameter, dtype: Any
) -> BoundKernelParameter:
    if param.kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise TypeError(
            f"parameter '{param.name}' uses unsupported parameter kind for TileLang DSL v1"
        )
    if param.default is not inspect._empty:
        raise TypeError(
            f"parameter '{param.name}' must not declare a default value in TileLang DSL v1"
        )
    if param.annotation is inspect._empty:
        raise TypeError(
            f"parameter '{param.name}' must declare a TileLang DSL type annotation"
        )

    annotation = param.annotation
    scalar_dtype = _validate_scalar_dtype(dtype, param.name)

    if annotation is TensorView:
        return BoundKernelParameter(
            name=param.name,
            kind="tensorview",
            annotation=annotation,
            dtype=scalar_dtype,
        )
    if annotation is Tile:
        return BoundKernelParameter(
            name=param.name,
            kind="tile",
            annotation=annotation,
            dtype=scalar_dtype,
        )
    if isinstance(annotation, ScalarType):
        if annotation != scalar_dtype:
            raise TypeError(
                f"scalar parameter '{param.name}' annotation {annotation!r} "
                f"does not match dtypes entry {scalar_dtype!r}"
            )
        return BoundKernelParameter(
            name=param.name,
            kind="scalar",
            annotation=annotation,
            dtype=scalar_dtype,
        )

    raise TypeError(
        f"parameter '{param.name}' uses unsupported annotation {annotation!r}"
    )


def _bind_parameters(
    py_fn: Callable[..., Any], dtypes: tuple[tuple[Any, ...], ...]
) -> tuple[BoundKernelParameter, ...]:
    if len(dtypes) != 1:
        raise ValueError(
            "TileLang DSL v1 requires dtypes to contain exactly one monomorphic signature tuple"
        )

    signature = inspect.signature(py_fn)
    params = tuple(signature.parameters.values())
    dtype_signature = dtypes[0]

    if len(dtype_signature) != len(params):
        raise ValueError(
            "single dtypes signature must match the decorated function parameter count"
        )

    return tuple(
        _bind_parameter(param, dtype)
        for param, dtype in zip(params, dtype_signature)
    )


def _build_descriptor(
    py_fn: Callable[..., Any],
    *,
    target: str,
    op: Any,
    dtypes: Any,
    name: Any,
    verify: Any,
) -> VKernelDescriptor:
    if not callable(py_fn):
        raise TypeError("@vkernel can only decorate callables")

    source_info = _load_function_source_info(py_fn)
    _validate_function_body(source_info)
    frozen_dtypes = _freeze_dtypes(dtypes)

    return VKernelDescriptor(
        target=_validate_target(target),
        op=_validate_op(op),
        dtypes=frozen_dtypes,
        name=_validate_name(py_fn, name),
        verify_enabled=_validate_verify(verify),
        parameters=_bind_parameters(py_fn, frozen_dtypes),
        _py_fn=py_fn,
        _source_info=source_info,
    )


def vkernel(
    py_fn: Callable[..., Any] | None = None,
    *,
    target: str = "a5",
    op: str | None = None,
    dtypes: Any = None,
    name: str | None = None,
    verify: bool = True,
    constraints: Any = _UNSET,
    priority: Any = _UNSET,
) -> VKernelDescriptor | Callable[[Callable[..., Any]], VKernelDescriptor]:
    """Create a TileLang DSL v1 kernel descriptor.

    v1 keeps only the minimal descriptor metadata surface:
    `target`, `op`, `dtypes`, `name`, and `verify`.
    """
    _reject_unsupported_decorator_feature("constraints", constraints)
    _reject_unsupported_decorator_feature("priority", priority)

    def wrap(fn: Callable[..., Any]) -> VKernelDescriptor:
        return _build_descriptor(
            fn,
            target=target,
            op=op,
            dtypes=dtypes,
            name=name,
            verify=verify,
        )

    if py_fn is None:
        return wrap
    return wrap(py_fn)


__all__ = [
    "BoundKernelParameter",
    "MaterializedMLIRModule",
    "TileLangFrontendError",
    "VKernelDescriptor",
    "vkernel",
]
