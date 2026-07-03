module attributes {pto.backend = "vpto", pto.target_arch = "a5"} {
  module attributes {pto.backend = "vpto", pto.kernel_kind = #pto.kernel_kind<vector>, pto.target_arch = "a5"} {
    func.func @gelu_poly_full_unroll_vfsim(%arg0: !pto.ptr<f32, gm>, %arg1: !pto.ptr<f32, gm>) attributes {pto.kernel} {
      %c64 = arith.constant 64 : index
      %c64_i32 = arith.constant 64 : i32
      %c256_i64 = arith.constant 256 : i64
      %c96_i64 = arith.constant 96 : i64
      %false = arith.constant false
      %c0 = arith.constant 0 : index
      %c147456_i64 = arith.constant 147456 : i64
      %c0_i64 = arith.constant 0 : i64
      %c1 = arith.constant 1 : index
      %c96 = arith.constant 96 : index
      %cst = arith.constant 5.000000e-01 : f32
      %cst_0 = arith.constant 0.707099974 : f32
      %cst_1 = arith.constant 3.920000e+00 : f32
      %cst_2 = arith.constant -3.920000e+00 : f32
      %cst_3 = arith.constant 5.344000e-01 : f32
      %cst_4 = arith.constant 7.551700e+00 : f32
      %cst_5 = arith.constant 101.62809 : f32
      %cst_6 = arith.constant 1393.80151 : f32
      %cst_7 = arith.constant 5063.7915 : f32
      %cst_8 = arith.constant 29639.3848 : f32
      %cst_9 = arith.constant 31.2128582 : f32
      %cst_10 = arith.constant 308.569641 : f32
      %cst_11 = arith.constant 3023.12476 : f32
      %cst_12 = arith.constant 14243.3662 : f32
      %cst_13 = arith.constant 26267.2246 : f32
      %cst_14 = arith.constant 1.000000e+00 : f32
      %0 = pto.castptr %c0_i64 : i64 -> !pto.ptr<f32, ub>
      %1 = pto.addptr %arg0, %c0 : <f32, gm> -> <f32, gm>
      %2 = pto.addptr %0, %c0 : <f32, ub> -> <f32, ub>
      pto.copy_gm_to_ubuf %1, %2, %c0_i64, %c96_i64, %c256_i64, %c0_i64, %c0_i64, %false, %c0_i64, %c256_i64, %c256_i64 : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i1, i64, i64, i64
      pto.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      pto.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %3 = pto.castptr %c147456_i64 : i64 -> !pto.ptr<f32, ub>
      pto.vecscope {
        scf.for %arg2 = %c0 to %c96 step %c1 {
          %mask, %scalar_out = pto.plt_b32 %c64_i32 : i32 -> !pto.mask<b32>, i32
          %6 = arith.muli %arg2, %c64 : index
          %7 = pto.addptr %0, %6 : <f32, ub> -> <f32, ub>
          %result = pto.vlds %7[%c0] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
          %8 = pto.vmuls %result, %cst, %mask : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>
          %9 = pto.vmuls %result, %cst_0, %mask : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>
          %10 = pto.vmins %9, %cst_1, %mask : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>
          %11 = pto.vmaxs %10, %cst_2, %mask : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>
          %12 = pto.vmul %11, %11, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          %13 = pto.vmuls %12, %cst_3, %mask : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>
          %14 = pto.vadds %13, %cst_4, %mask : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>
          %15 = pto.vmul %14, %12, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          %16 = pto.vadds %15, %cst_5, %mask : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>
          %17 = pto.vmul %16, %12, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          %18 = pto.vadds %17, %cst_6, %mask : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>
          %19 = pto.vmul %18, %12, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          %20 = pto.vadds %19, %cst_7, %mask : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>
          %21 = pto.vmul %20, %12, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          %22 = pto.vadds %21, %cst_8, %mask : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>
          %23 = pto.vmul %22, %11, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          %24 = pto.vadds %12, %cst_9, %mask : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>
          %25 = pto.vmul %24, %12, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          %26 = pto.vadds %25, %cst_10, %mask : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>
          %27 = pto.vmul %26, %12, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          %28 = pto.vadds %27, %cst_11, %mask : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>
          %29 = pto.vmul %28, %12, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          %30 = pto.vadds %29, %cst_12, %mask : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>
          %31 = pto.vmul %30, %12, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          %32 = pto.vadds %31, %cst_13, %mask : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>
          %33 = pto.vdiv %23, %32, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          %34 = pto.vadds %33, %cst_14, %mask : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>
          %35 = pto.vmul %34, %8, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          %36 = pto.addptr %3, %6 : <f32, ub> -> <f32, ub>
          pto.vsts %35, %36[%c0], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
        }
      }
      %4 = pto.addptr %3, %c0 : <f32, ub> -> <f32, ub>
      %5 = pto.addptr %arg1, %c0 : <f32, gm> -> <f32, gm>
      pto.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      pto.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      pto.copy_ubuf_to_gm %4, %5, %c0_i64, %c96_i64, %c256_i64, %c0_i64, %c256_i64, %c256_i64 : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64
      pto.barrier <PIPE_ALL> {pto.auto_sync_tail_barrier}
      return
    }
  }
}

