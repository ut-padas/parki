"""Ensure kernel dispatch does not need PyKokkos argument type inference."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pykokkos.core.type_inference.args_type_inference import get_annotations

from parkipy.ewald._prepare import Options

KERNEL_ROOT = Path(__file__).resolve().parents[1] / "parkipy"
KERNEL_FILES = [
    KERNEL_ROOT / "_pk_kernels/_celllist.py",
    KERNEL_ROOT / "_pk_kernels/templates/_celllist_template.py",
    *sorted(KERNEL_ROOT.glob("ewald/_pk_kernels/*_workunits.py")),
    *sorted(KERNEL_ROOT.glob("ewald/_pk_kernels/templates/*_workunits.py")),
]


@pytest.mark.parametrize(
    "path", KERNEL_FILES, ids=lambda path: str(path.relative_to(KERNEL_ROOT))
)
def test_workunits_skip_argument_inference(path):
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.FunctionDef):
            continue
        decorators = [
            decorator.func if isinstance(decorator, ast.Call) else decorator
            for decorator in node.decorator_list
        ]
        if not any(
            isinstance(decorator, ast.Attribute) and decorator.attr == "workunit"
            for decorator in decorators
        ):
            continue

        # Deliberately uninferable arguments: complete annotations must make
        # PyKokkos return before inspecting either the policy or argument values.
        arguments = {arg.arg: object() for arg in node.args.args}
        assert (
            get_annotations("parallel_for", (None, node), object(), arguments) is None
        ), f"{path}:{node.lineno}: {node.name} requires argument inference"


@pytest.mark.parametrize("upsampling", [[1, 1, 1], [1.0, 2.5, 2.5]])
def test_integer_box_and_upsampling_match_view_annotations(upsampling):
    params = SimpleNamespace(
        periodicity=3,
        box=[1, 1, 1],
        xi=1.0,
        rc=0.25,
        h=0.1,
        box_off=[0, 0, 0],
        grid_res=10.0,
        grid_shape_ext=[10, 10, 10],
        window_P=4,
        actual_upsampling=upsampling,
        distributed=False,
        glb_box=[1, 1, 1],
        glb_box_ext=[1, 1, 1],
        glb_grid_shape_ext=[10, 10, 10],
    )
    options = Options.from_params(params, "CPU")
    assert options.actual_upsampling_global.dtype == np.float64
    assert options.glb_box_ext.dtype == np.float64
    np.testing.assert_array_equal(options.actual_upsampling_global, upsampling)
    np.testing.assert_array_equal(options.glb_box_ext, params.glb_box_ext)
