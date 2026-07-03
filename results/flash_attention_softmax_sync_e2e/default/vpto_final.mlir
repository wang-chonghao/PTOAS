module attributes {pto.backend = "vpto", pto.target_arch = "a5"} {
  module attributes {pto.backend = "vpto", pto.kernel_kind = #pto.kernel_kind<vector>, pto.target_arch = "a5"} {
    func.func @flash_attention_softmax_block(%arg0: !pto.ptr<f32, gm>, %arg1: !pto.ptr<f32, gm>) attributes {pto.kernel} {
      %c32_i32 = arith.constant 32 : i32
      %c128_i64 = arith.constant 128 : i64
      %c32_i64 = arith.constant 32 : i64
      %false = arith.constant false
      %c0 = arith.constant 0 : index
      %c16384_i64 = arith.constant 16384 : i64
      %c0_i64 = arith.constant 0 : i64
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %cst = arith.constant 0.176776707 : f32
      %cst_0 = arith.constant -4.000100e+00 : f32
      %cst_1 = arith.constant 4.000100e+00 : f32
      %cst_2 = arith.constant 1.000100e+00 : f32
      %cst_3 = arith.constant 3.200010e+01 : f32
      %cst_4 = arith.constant 5.000000e-01 : f32
      %cst_5 = arith.constant 0.166666701 : f32
      %cst_6 = arith.constant 0.0416666716 : f32
      %0 = pto.castptr %c0_i64 : i64 -> !pto.ptr<f32, ub>
      %1 = pto.addptr %arg0, %c0 : <f32, gm> -> <f32, gm>
      %2 = pto.addptr %0, %c0 : <f32, ub> -> <f32, ub>
      pto.copy_gm_to_ubuf %1, %2, %c0_i64, %c32_i64, %c128_i64, %c0_i64, %c0_i64, %false, %c0_i64, %c128_i64, %c128_i64 : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i1, i64, i64, i64
      pto.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      pto.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      %3 = pto.castptr %c16384_i64 : i64 -> !pto.ptr<f32, ub>
      pto.vecscope {
        scf.for %arg2 = %c0 to %c32 step %c1 {
          %mask, %scalar_out = pto.plt_b32 %c32_i32 : i32 -> !pto.mask<b32>, i32
          %6 = arith.muli %arg2, %c32 : index
          %7 = pto.addptr %0, %6 : <f32, ub> -> <f32, ub>
          %result = pto.vlds %7[%c0] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
          %8 = pto.vmuls %result, %cst, %mask : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>
          %9 = pto.vmaxs %8, %cst_0, %mask : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>
          %10 = pto.vmins %9, %cst_1, %mask : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>
          %11 = pto.vmul %10, %10, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          %12 = pto.vmul %11, %10, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          %13 = pto.vmul %12, %10, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          %14 = pto.vmuls %11, %cst_4, %mask : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>
          %15 = pto.vmuls %12, %cst_5, %mask : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>
          %16 = pto.vmuls %13, %cst_6, %mask : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>
          %17 = pto.vadd %10, %14, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          %18 = pto.addptr %3, %6 : <f32, ub> -> <f32, ub>
          %19 = pto.vadd %17, %15, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          %20 = pto.vadd %19, %16, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          %21 = pto.vadds %20, %cst_2, %mask : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>
          %22 = pto.vbr %cst_3 : f32 -> !pto.vreg<64xf32>
          %23 = pto.vdiv %21, %22, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          pto.vsts %23, %18[%c0], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
        }
      }
      %4 = pto.addptr %3, %c0 : <f32, ub> -> <f32, ub>
      %5 = pto.addptr %arg1, %c0 : <f32, gm> -> <f32, gm>
      pto.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      pto.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      pto.copy_ubuf_to_gm %4, %5, %c0_i64, %c32_i64, %c128_i64, %c0_i64, %c128_i64, %c128_i64 : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64
      pto.barrier <PIPE_ALL> {pto.auto_sync_tail_barrier}
      return
    }
  }
}

