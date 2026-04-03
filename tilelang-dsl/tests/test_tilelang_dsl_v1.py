import tempfile
import unittest
from importlib import util
from pathlib import Path

import tilelang_dsl as pto
from tilelang_dsl.frontend_ast import build_frontend_kernel_node
from tilelang_dsl.lowering import AuthoringModule, lower_semantic_kernel
from tilelang_dsl.semantic import (
    SemanticAssignStmt,
    SemanticCallExpr,
    SemanticDmaLoadStmt,
    SemanticDmaStoreStmt,
    SemanticForStmt,
    SemanticIfStmt,
    SemanticIndexType,
    SemanticMaskType,
    SemanticPipeBarrierStmt,
    SemanticScalarType,
    SemanticSetFlagStmt,
    SemanticStrictVecscopeStmt,
    SemanticTensorViewType,
    SemanticTileType,
    SemanticVectorStoreStmt,
    SemanticWaitFlagStmt,
    analyze_frontend_kernel,
)


class TileLangDSLPackageTests(unittest.TestCase):
    def test_package_exports_surface(self) -> None:
        self.assertIsNotNone(pto.__file__)
        self.assertTrue(hasattr(pto, "vkernel"))
        self.assertTrue(hasattr(pto, "TensorView"))
        self.assertTrue(hasattr(pto, "Tile"))
        self.assertTrue(hasattr(pto, "TileSpecialization"))
        self.assertTrue(hasattr(pto, "PAT"))
        self.assertTrue(hasattr(pto, "PIPE"))
        self.assertTrue(hasattr(pto, "EVENT"))


class TileLangDSLDescriptorTests(unittest.TestCase):
    def test_descriptor_metadata_and_parameter_binding(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f16, pto.i32)], verify=False)
        def kernel(inp: pto.TensorView, tile: pto.Tile, scale: pto.i32):
            return None

        self.assertEqual(kernel.target, "a5")
        self.assertEqual(kernel.op, "eltwise")
        self.assertEqual(kernel.name, "kernel")
        self.assertFalse(kernel.verify_enabled)
        self.assertEqual(kernel.metadata["verify"], False)
        self.assertEqual(kernel.dtype_signature, (pto.f32, pto.f16, pto.i32))
        self.assertEqual(
            [(param.name, param.kind, param.dtype) for param in kernel.parameters],
            [("inp", "tensorview", pto.f32), ("tile", "tile", pto.f16), ("scale", "scalar", pto.i32)],
        )
        self.assertEqual(kernel.parameters[0].element_dtype, pto.f32)
        self.assertEqual(kernel.parameters[1].element_dtype, pto.f16)
        self.assertIsNone(kernel.parameters[2].element_dtype)

    def test_specialization_enables_materialization_apis(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f16)])
        def kernel(inp: pto.TensorView, tile: pto.Tile):
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(16, 32),
                memory_space=pto.MemorySpace.UB,
                config=pto.TileConfig.from_mapping({"layout": "row_major"}),
            )
        )

        self.assertIn("tile", specialized.specializations_by_name)
        text = specialized.mlir_text()
        self.assertIn("// tilelang.target = a5", text)
        self.assertIn("// tilelang.specialize tile shape=(16, 32) memory_space=ub", text)
        self.assertIn('module attributes {pto.target_arch = "a5"} {', text)
        self.assertIn("func.func @kernel(%arg0: !pto.ptr<f32, gm>, %arg1: !pto.ptr<f16, ub>) {", text)
        module = specialized.mlir_module()
        self.assertEqual(type(module).__name__, "MaterializedMLIRModule")
        self.assertTrue(module.verify())
        self.assertTrue(specialized.verify())

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "kernel.mlir"
            specialized.emit(out)
            self.assertEqual(out.read_text(encoding="utf-8"), text)

    def test_descriptor_materialization_flows_through_pipeline(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f16, pto.i32)])
        def kernel(inp: pto.TensorView, tile: pto.Tile, scale: pto.i32):
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(8, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        frontend_kernel = build_frontend_kernel_node(specialized)
        self.assertEqual(frontend_kernel.name, "kernel")
        self.assertEqual(
            [(param.name, param.kind) for param in frontend_kernel.parameters],
            [("inp", "tensorview"), ("tile", "tile"), ("scale", "scalar")],
        )
        self.assertEqual(frontend_kernel.tile_specializations[0].shape, (8, 16))

        semantic_kernel = analyze_frontend_kernel(frontend_kernel)
        self.assertEqual(semantic_kernel.symbol_name, "kernel")
        self.assertEqual(semantic_kernel.tile_bindings[0].memory_space, "ub")

        authoring_module = lower_semantic_kernel(semantic_kernel)
        self.assertIsInstance(authoring_module, AuthoringModule)
        self.assertEqual(authoring_module.render(), specialized.mlir_text())
        self.assertIn("return", authoring_module.render())

    def test_semantic_pipeline_binds_parameter_loop_and_strict_vecscope_types(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f16, pto.i32)])
        def kernel(inp: pto.TensorView, tile: pto.Tile, scale: pto.i32):
            rows = tile.shape[0]
            step = rows
            with pto.strict_vecscope(inp, tile, scale, 0, rows, step) as (
                vin,
                vtmp,
                factor,
                lb,
                ub,
                vec_step,
            ):
                for lane in range(lb, ub, vec_step):
                    current = factor
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(8, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        frontend_kernel = build_frontend_kernel_node(specialized)
        self.assertEqual(len(frontend_kernel.body), 4)

        semantic_kernel = analyze_frontend_kernel(frontend_kernel)
        self.assertIsInstance(semantic_kernel.parameters[0].type, SemanticTensorViewType)
        self.assertIsInstance(semantic_kernel.parameters[1].type, SemanticTileType)
        self.assertEqual(semantic_kernel.parameters[1].type.shape, (8, 16))
        self.assertIsInstance(semantic_kernel.parameters[2].type, SemanticScalarType)

        rows_assign = semantic_kernel.body[0]
        self.assertIsInstance(rows_assign, SemanticAssignStmt)
        self.assertIsInstance(rows_assign.targets[0].type, SemanticIndexType)
        self.assertTrue(rows_assign.targets[0].ssa_name.startswith("%rows_"))

        vecscope_stmt = semantic_kernel.body[2]
        self.assertIsInstance(vecscope_stmt, SemanticStrictVecscopeStmt)
        self.assertEqual(
            [binding.name for binding in vecscope_stmt.block_arguments],
            ["vin", "vtmp", "factor", "lb", "ub", "vec_step"],
        )
        self.assertIsInstance(vecscope_stmt.block_arguments[0].type, SemanticTensorViewType)
        self.assertIsInstance(vecscope_stmt.block_arguments[1].type, SemanticTileType)
        self.assertIsInstance(vecscope_stmt.block_arguments[2].type, SemanticScalarType)
        self.assertIsInstance(vecscope_stmt.block_arguments[3].type, SemanticIndexType)
        self.assertIsInstance(vecscope_stmt.block_arguments[4].type, SemanticIndexType)
        self.assertIsInstance(vecscope_stmt.block_arguments[5].type, SemanticIndexType)
        self.assertTrue(vecscope_stmt.block_arguments[0].ssa_name.startswith("%vin_"))

        loop_stmt = vecscope_stmt.body[0]
        self.assertIsInstance(loop_stmt, SemanticForStmt)
        self.assertEqual(loop_stmt.induction_variable.name, "lane")
        self.assertIsInstance(loop_stmt.induction_variable.type, SemanticIndexType)
        self.assertTrue(loop_stmt.induction_variable.ssa_name.startswith("%lane_"))
        self.assertEqual(loop_stmt.loop_carried, ())

        text = specialized.mlir_text()
        self.assertIn("%rows_", text)
        self.assertIn("= arith.constant 8 : index", text)
        self.assertIn("pto.strict_vecscope(%arg0, %arg1, %arg2, %c0, %rows_", text)
        self.assertIn("^bb0(", text)
        self.assertIn("scf.for %lane_", text)
        self.assertIn("to %ub_6 step %vec_step_7 {", text)

    def test_dma_load_and_store_lower_to_dma_programming_and_copy_ops(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32, pto.f32)])
        def kernel(inp: pto.TensorView, out: pto.TensorView, tile: pto.Tile):
            pto.dma_load(inp[0:16, 0:16], tile)
            pto.dma_store(tile, out[0:16, 0:16])
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(16, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertIsInstance(semantic_kernel.body[0], SemanticDmaLoadStmt)
        self.assertIsInstance(semantic_kernel.body[1], SemanticDmaStoreStmt)

        text = specialized.mlir_text()
        self.assertIn(
            "func.func @kernel(%arg0: !pto.ptr<f32, gm>, %arg1: !pto.ptr<f32, gm>, %arg2: !pto.ptr<f32, ub>) {",
            text,
        )
        self.assertIn("pto.set_loop_size_outtoub %c1_i64, %c1_i64 : i64, i64", text)
        self.assertIn(
            "pto.copy_gm_to_ubuf %arg0, %arg2, %c0_i64, %c16_i64, %c64_i64, %c0_i64, %c0_i64, %false, %c0_i64, %c64_i64, %c64_i64",
            text,
        )
        self.assertIn("pto.set_loop_size_ubtoout %c1_i64, %c1_i64 : i64, i64", text)
        self.assertIn(
            "pto.copy_ubuf_to_gm %arg2, %arg1, %c0_i64, %c16_i64, %c64_i64, %c0_i64, %c64_i64, %c64_i64",
            text,
        )

    def test_make_mask_vlds_vsts_and_vector_families_lower_inside_strict_vecscope(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32, pto.f32)])
        def kernel(inp: pto.TensorView, tile: pto.Tile, scale: pto.f32):
            pto.dma_load(inp[0:16, 0:16], tile)
            with pto.strict_vecscope(tile, tile, scale, 0, 256, 64) as (
                src,
                dst,
                factor,
                lb,
                ub,
                step,
            ):
                for lane in range(lb, ub, step):
                    mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                    vec = pto.vlds(src, lane)
                    biased = pto.vadds(vec, factor, mask)
                    summed = pto.vadd(biased, vec, mask)
                    activated = pto.vrelu(summed, mask)
                    pto.vsts(activated, dst, lane, mask)
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(16, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope = semantic_kernel.body[1]
        self.assertIsInstance(vecscope, SemanticStrictVecscopeStmt)
        loop_stmt = vecscope.body[0]
        self.assertIsInstance(loop_stmt, SemanticForStmt)
        mask_assign = loop_stmt.body[0]
        self.assertIsInstance(mask_assign, SemanticAssignStmt)
        self.assertIsInstance(mask_assign.value, SemanticCallExpr)
        self.assertEqual(mask_assign.value.name, "make_mask")
        self.assertIsInstance(mask_assign.targets[0].type, SemanticMaskType)
        self.assertIsInstance(loop_stmt.body[-1], SemanticVectorStoreStmt)

        text = specialized.mlir_text()
        self.assertIn('%mask_7 = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>', text)
        self.assertIn("%vec_8 = pto.vlds %src_0[%lane_6] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>", text)
        self.assertIn(
            "%biased_9 = pto.vadds %vec_8, %factor_2, %mask_7 : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>",
            text,
        )
        self.assertIn(
            "%summed_10 = pto.vadd %biased_9, %vec_8, %mask_7 : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>",
            text,
        )
        self.assertIn(
            "%activated_11 = pto.vrelu %summed_10, %mask_7 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>",
            text,
        )
        self.assertIn(
            "pto.vsts %activated_11, %dst_1[%lane_6], %mask_7 : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>",
            text,
        )

    def test_if_else_and_sync_ops_lower_to_scf_if_and_authoring_sync_ops(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32, pto.i32)])
        def kernel(inp: pto.TensorView, tile: pto.Tile, flag: pto.i32):
            pto.set_flag(pto.PIPE.MTE2, pto.PIPE.V, pto.EVENT.ID0)
            pto.wait_flag(pto.PIPE.MTE2, pto.PIPE.V, pto.EVENT.ID0)
            step = 64
            if flag:
                step = 64
                pto.set_flag(pto.PIPE.V, pto.PIPE.MTE3, pto.EVENT.ID0)
            else:
                step = 128
                pto.wait_flag(pto.PIPE.V, pto.PIPE.MTE3, pto.EVENT.ID0)
            with pto.strict_vecscope(tile, tile, 0, 256, step) as (src, dst, lb, ub, vec_step):
                for lane in range(lb, ub, vec_step):
                    mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                    vec = pto.vlds(src, lane)
                    pto.vsts(vec, dst, lane, mask)
            pto.pipe_barrier(pto.PIPE.ALL)
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(16, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertIsInstance(semantic_kernel.body[0], SemanticSetFlagStmt)
        self.assertIsInstance(semantic_kernel.body[1], SemanticWaitFlagStmt)
        self.assertIsInstance(semantic_kernel.body[3], SemanticIfStmt)
        self.assertIsInstance(semantic_kernel.body[5], SemanticPipeBarrierStmt)

        text = specialized.mlir_text()
        self.assertIn('pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]', text)
        self.assertIn('pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]', text)
        self.assertIn("= arith.cmpi ne, %arg2, %c0_i32 : i32", text)
        self.assertIn("%step_3 = scf.if %tmp_0 -> (index) {", text)
        self.assertIn('pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]', text)
        self.assertIn('pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]', text)
        self.assertRegex(text, r"scf\.yield %step_\d+ : index")
        self.assertIn("%step_2 = arith.constant 128 : index", text)
        self.assertIn("pto.strict_vecscope(%arg1, %arg1, %c0, %c256, %step_3)", text)
        self.assertIn("scf.for %lane_", text)
        self.assertIn("pto.barrier #pto.pipe<PIPE_ALL>", text)

    def test_strict_vecscope_rejects_implicit_capture_during_semantic_analysis(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f16, pto.i32)])
        def kernel(inp: pto.TensorView, tile: pto.Tile, scale: pto.i32):
            with pto.strict_vecscope(inp, tile) as (vin, vtmp):
                leaked = scale
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(8, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        with self.assertRaises(ValueError) as ctx:
            analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertIn("implicit capture of 'scale' is not allowed", str(ctx.exception))


class TileLangDSLDiagnosticsTests(unittest.TestCase):
    def test_matcher_feature_diagnostics_point_to_follow_up_change(self) -> None:
        cases = [
            lambda: pto.vkernel(op="x", dtypes=[(pto.f32,)], constraints=[])(lambda x: None),
            lambda: pto.vkernel(op="x", dtypes=[(pto.f32,)], priority=1)(lambda x: None),
            lambda: pto.vkernel(op="x", dtypes=[(pto.f32,), (pto.f16,)])(lambda x: None),
            lambda: pto.vkernel(op="x", dtypes=[(pto.AnyFloat,)])(lambda x: None),
            lambda: pto.vkernel(op="x", dtypes=[(pto.TypeVar("T"),)])(lambda x: None),
        ]

        for thunk in cases:
            with self.assertRaises(ValueError) as ctx:
                thunk()
            self.assertIn(
                "extend-tilelang-dsl-matcher-and-advanced-surface",
                str(ctx.exception),
            )

    def test_unsupported_python_syntax_reports_source_location(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="x", dtypes=[(pto.f32,)])
            def kernel(x: pto.TensorView):
                while True:
                    return None

        self.assertIn("unsupported Python syntax `while`", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_arbitrary_external_call_reports_source_location(self) -> None:
        def helper():
            return None

        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="x", dtypes=[(pto.f32,)])
            def kernel(x: pto.TensorView):
                helper()
                return None

        self.assertIn("arbitrary external call `helper`", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_unsupported_pto_surface_reports_source_location(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="x", dtypes=[(pto.f32,)])
            def kernel(x: pto.TensorView):
                pto.vadd(x)
                return None

        self.assertIn("vector op surface `pto.vadd` requires explicit pto.strict_vecscope", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_missing_specialization_reports_source_location(self) -> None:
        @pto.vkernel(op="x", dtypes=[(pto.f32, pto.f16)])
        def kernel(x: pto.TensorView, tile: pto.Tile):
            return None

        with self.assertRaises(pto.TileLangFrontendError) as ctx:
            kernel.mlir_text()

        self.assertIn("requires specialize() bindings for bare Tile parameters", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_dynamic_shape_and_illegal_profile_report_source_location(self) -> None:
        @pto.vkernel(op="x", dtypes=[(pto.f32, pto.f16)])
        def kernel(x: pto.TensorView, tile: pto.Tile):
            return None

        with self.assertRaises(pto.TileLangFrontendError) as dynamic_ctx:
            kernel.specialize(tile={"shape": (16, "n"), "memory_space": "ub"})
        self.assertIn("dynamic physical Tile shape is not supported", str(dynamic_ctx.exception))
        self.assertIn(f"{__file__}:", str(dynamic_ctx.exception))

        with self.assertRaises(pto.TileLangFrontendError) as rank_ctx:
            kernel.specialize(tile={"shape": (4, 4, 4), "memory_space": "ub"})
        self.assertIn("v1 only supports rank-1 or rank-2 Tile shapes", str(rank_ctx.exception))
        self.assertIn(f"{__file__}:", str(rank_ctx.exception))

        with self.assertRaises(pto.TileLangFrontendError) as space_ctx:
            kernel.specialize(tile={"shape": (4, 4), "memory_space": "gm"})
        self.assertIn("v1 only supports MemorySpace.UB", str(space_ctx.exception))
        self.assertIn(f"{__file__}:", str(space_ctx.exception))


if __name__ == "__main__":
    unittest.main()
