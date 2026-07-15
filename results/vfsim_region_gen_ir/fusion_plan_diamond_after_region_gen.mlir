// -----// IR Dump After PTOFusionRegionGen (pto-fusion-region-gen) //----- //
func.func @fusion_plan_diamond(%arg0: !pto.tile_buf<vec, 32x32xf32>, %arg1: !pto.tile_buf<vec, 32x32xf32>, %arg2: !pto.tile_buf<vec, 32x32xf32>, %arg3: !pto.tile_buf<vec, 32x32xf32>, %arg4: !pto.tile_buf<vec, 32x32xf32>, %arg5: !pto.tile_buf<vec, 32x32xf32>) {
  pto.fusion_region {
    %0 = pto.alloc_tile : !pto.tile_buf<vec, 32x32xf32>
    %1 = pto.alloc_tile : !pto.tile_buf<vec, 32x32xf32>
    %2 = pto.alloc_tile : !pto.tile_buf<vec, 32x32xf32>
    %3 = pto.alloc_tile : !pto.tile_buf<vec, 32x32xf32>
    %4 = pto.alloc_tile : !pto.tile_buf<vec, 32x32xf32>
    %5 = pto.alloc_tile : !pto.tile_buf<vec, 32x32xf32>
    %6 = pto.alloc_tile : !pto.tile_buf<vec, 32x32xf32>
    %7 = pto.alloc_tile : !pto.tile_buf<vec, 32x32xf32>
    pto.tmax ins(%arg0, %arg1 : !pto.tile_buf<vec, 32x32xf32>, !pto.tile_buf<vec, 32x32xf32>) outs(%0 : !pto.tile_buf<vec, 32x32xf32>)
    pto.tsub ins(%0, %arg2 : !pto.tile_buf<vec, 32x32xf32>, !pto.tile_buf<vec, 32x32xf32>) outs(%1 : !pto.tile_buf<vec, 32x32xf32>)
    pto.tsub ins(%0, %arg3 : !pto.tile_buf<vec, 32x32xf32>, !pto.tile_buf<vec, 32x32xf32>) outs(%2 : !pto.tile_buf<vec, 32x32xf32>)
    pto.texp ins(%1 : !pto.tile_buf<vec, 32x32xf32>) outs(%3 : !pto.tile_buf<vec, 32x32xf32>)
    pto.texp ins(%2 : !pto.tile_buf<vec, 32x32xf32>) outs(%4 : !pto.tile_buf<vec, 32x32xf32>)
    pto.tmul ins(%3, %arg4 : !pto.tile_buf<vec, 32x32xf32>, !pto.tile_buf<vec, 32x32xf32>) outs(%5 : !pto.tile_buf<vec, 32x32xf32>)
    pto.tmul ins(%4, %arg5 : !pto.tile_buf<vec, 32x32xf32>, !pto.tile_buf<vec, 32x32xf32>) outs(%6 : !pto.tile_buf<vec, 32x32xf32>)
    pto.tadd ins(%5, %6 : !pto.tile_buf<vec, 32x32xf32>, !pto.tile_buf<vec, 32x32xf32>) outs(%7 : !pto.tile_buf<vec, 32x32xf32>)
    pto.yield() : () -> ()
  } {pto.fusion.group_id = 0 : i64, pto.fusion.unroll = 1 : i64} : 
  return
}

