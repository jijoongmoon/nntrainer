#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TSM 64
#define TSN 128
#define TSK 16
#define WPTM 4
#define WPTN 8
#define RTSM 16
#define RTSN 16

#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#define ALIGN(a, b) (CEIL_DIV(a, b) * (b))



__attribute__((qcom_reqd_sub_group_size("full"))) kernel void
fully_connected_gpu_int4_gemm_adreno(__global half *input, const __global half *scales,
                              __global half *output,
                              const __global char *weights, const int K,
                              const int N,
                              const int M,
                              const int quantization_group_size) {
    const int ALIGN_K = ALIGN(K, quantization_group_size);
    const int align_N = ALIGN(N, 32);

    const int m = get_global_id(0);
    const int n = get_global_id(1);
    const int offsetM = TSM * (m/RTSM);
    const int offsetN = TSN * (n/RTSN);

    const int tile_idm = get_local_id(0);
    const int tile_idn = get_local_id(1);

    __local half input_sub[TSK][TSM];
    __local half weights_sub[TSK][TSN];

    half4 weights_reg;
    half input_reg[WPTM];

    half acc[WPTN][WPTM];

    int numTiles = CEIL_DIV(K,TSK);
    int max_linput = (TSK*TSM)/(4*RTSM*RTSN);
    int max_lweight = (TSK*TSN)/(8*RTSM*RTSN);
    # pragma unroll
    for (int t=0; t<numTiles; t++) {
        # pragma unroll
        for (int la=0; la<max_linput; la++) {
            int tile_id = tile_idn * RTSM + tile_idm;
            int id = 4*la * RTSN * RTSM + 4 * tile_id;
            int x = id % TSK;
            int y = id / TSK;
            int input_col = TSK * t + x;
            int input_row = offsetM + y;
            if ((input_col<K)&&(input_row<M)){
                half4 temp;
                temp = vload4(0,input + ALIGN_K * input_row + input_col);

                input_sub[input_col%TSK][input_row%TSM] = temp.s0;
                input_sub[input_col%TSK+1][input_row%TSM] = temp.s1;
                input_sub[input_col%TSK+2][input_row%TSM] = temp.s2;
                input_sub[input_col%TSK+3][input_row%TSM] = temp.s3;
            }
            else{
                input_sub[input_col%TSK][input_row%TSM] = 0;
                input_sub[input_col%TSK+1][input_row%TSM] = 0;
                input_sub[input_col%TSK+2][input_row%TSM] = 0;
                input_sub[input_col%TSK+3][input_row%TSM] = 0;
            }
        }
        # pragma unroll
        for (int la=0; la<max_lweight; la++) {
            int tile_id = tile_idn * RTSM + tile_idm;
            int id = 4 * la * RTSN * RTSM + 4 * tile_id;
            int x = id % TSN;
            int y = id / TSN;
            int weights_col = offsetN + x;
            int weights_row = TSK * t + 2*y;
            
            if ((weights_col<N)&&(weights_row<K)) {
                char4 packed_w;
                const int index = (weights_col / 32) * (16 * ALIGN_K) + (weights_col % 32) + (weights_row / 2) * 32;
                packed_w = vload4(0,weights+index);

                char4 w_lo;
                char4 w_hi;

                w_lo.s0 = packed_w.s0 & 0xF;
                w_hi.s0 = packed_w.s0 >> 4;
                if (w_lo.s0 > 7) w_lo.s0 -= 16;
                if (w_hi.s0 > 7) w_hi.s0 -= 16;

                w_lo.s1 = packed_w.s1 & 0xF;
                w_hi.s1 = packed_w.s1 >> 4;
                if (w_lo.s1 > 7) w_lo.s1 -= 16;
                if (w_hi.s1 > 7) w_hi.s1 -= 16;

                w_lo.s2 = packed_w.s2 & 0xF;
                w_hi.s2 = packed_w.s2 >> 4;
                if (w_lo.s2 > 7) w_lo.s2 -= 16;
                if (w_hi.s2 > 7) w_hi.s2 -= 16;

                w_lo.s3 = packed_w.s3 & 0xF;
                w_hi.s3 = packed_w.s3 >> 4;
                if (w_lo.s3 > 7) w_lo.s3 -= 16;
                if (w_hi.s3 > 7) w_hi.s3 -= 16;

                half4 scale;
                scale = vload4(0,scales + (weights_row/quantization_group_size)*align_N + weights_col);

                weights_sub[weights_row%TSK][weights_col%TSN] = w_lo.s0 * scale.s0;
                weights_sub[weights_row%TSK+1][weights_col%TSN] = w_hi.s0 * scale.s0;

                weights_sub[weights_row%TSK][weights_col%TSN+1] = w_lo.s1 * scale.s1;
                weights_sub[weights_row%TSK+1][weights_col%TSN+1] = w_hi.s1 * scale.s1;

                weights_sub[weights_row%TSK][weights_col%TSN+2] = w_lo.s2 * scale.s2;
                weights_sub[weights_row%TSK+1][weights_col%TSN+2] = w_hi.s2 * scale.s2;

                weights_sub[weights_row%TSK][weights_col%TSN+3] = w_lo.s3 * scale.s3;
                weights_sub[weights_row%TSK+1][weights_col%TSN+3] = w_hi.s3 * scale.s3;
            }
            else{
                weights_sub[weights_row%TSK][weights_col%TSN] = 0;
                weights_sub[weights_row%TSK+1][weights_col%TSN] = 0;

                weights_sub[weights_row%TSK][weights_col%TSN+1] = 0;
                weights_sub[weights_row%TSK+1][weights_col%TSN+1] = 0;

                weights_sub[weights_row%TSK][weights_col%TSN+2] = 0;
                weights_sub[weights_row%TSK+1][weights_col%TSN+2] = 0;

                weights_sub[weights_row%TSK][weights_col%TSN+3] = 0;
                weights_sub[weights_row%TSK+1][weights_col%TSN+3] = 0;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        # pragma unroll
        for (int k=0;k<TSK;k++){

            # pragma unroll
            for (int wm=0; wm<WPTM/4; wm++) {
                int row = 4*tile_idm + wm*RTSM*4;
                half4 temp;

                temp = vload4(0,input_sub[k] + row);

                input_reg[4*wm] = temp.s0;
                input_reg[4*wm+1] = temp.s1;
                input_reg[4*wm+2] = temp.s2;
                input_reg[4*wm+3] = temp.s3;
            }
            # pragma unroll
            for (int wn=0; wn<WPTN/4; wn++) {
                int col = 4*tile_idn + 4*wn*RTSN;

                weights_reg = vload4(0,weights_sub[k] + col);

                # pragma unroll
                for (int wm=0; wm<WPTM; wm++) {
                    acc[4*wn][wm] += weights_reg.s0 * input_reg[wm];
                    acc[4*wn+1][wm] += weights_reg.s1 * input_reg[wm];
                    acc[4*wn+2][wm] += weights_reg.s2 * input_reg[wm];
                    acc[4*wn+3][wm] += weights_reg.s3 * input_reg[wm];
                }
            }

        }

        barrier(CLK_LOCAL_MEM_FENCE);
        
    }
    # pragma unroll
    for (int wm=0; wm<WPTM/4; wm++) {
        int globalRow = offsetM + 4*tile_idm + wm*RTSM*4;
        # pragma unroll
        for (int wn=0; wn<WPTN/4; wn++) {
            int globalCol = offsetN + 4*tile_idn + 4*wn*RTSN;

            if ((globalRow<M)&&(globalCol<N)){
                # pragma unroll
                for (int i=0;i<4;i++){
                    # pragma unroll
                    for (int j=0;j<4;j++){
                        output[(globalRow+i)*N + globalCol+j] = acc[4*wn+j][4*wm+i];
                    }
                }
            }
        }
    }
}







