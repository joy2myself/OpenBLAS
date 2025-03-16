#include "common.h"

#if !defined(DOUBLE)
#define VSETVL(n)       __riscv_vsetvl_e32m4(n)
#define FLOAT_V_T       vfloat32m4_t
#define VLSEV_FLOAT     __riscv_vlse32_v_f32m4
#define VSSEV_FLOAT     __riscv_vsse32_v_f32m4
#define VFMACCVV_FLOAT  __riscv_vfmacc_vv_f32m4
#define VFNMSACVV_FLOAT __riscv_vfnmsac_vv_f32m4
#define VFMVVF_FLOAT    __riscv_vfmv_v_f_f32m4
#define VFREDSUM_FLOAT(vs, init, gvl) \
    __riscv_vfmv_f_s_f32m1_f32( \
        __riscv_vfredusum_vs_f32m4_f32m1(vs, __riscv_vfmv_v_f_f32m1(init, gvl), gvl) \
    )
#else
#define VSETVL(n)       __riscv_vsetvl_e64m4(n)
#define FLOAT_V_T       vfloat64m4_t
#define VLSEV_FLOAT     __riscv_vlse64_v_f64m4
#define VSSEV_FLOAT     __riscv_vsse64_v_f64m4
#define VFMACCVV_FLOAT  __riscv_vfmacc_vv_f64m4
#define VFNMSACVV_FLOAT __riscv_vfnmsac_vv_f64m4
#define VFMVVF_FLOAT    __riscv_vfmv_v_f_f64m4
#define VFREDSUM_FLOAT(vs, init, gvl) \
    __riscv_vfmv_f_s_f64m1_f64( \
        __riscv_vfredusum_vs_f64m4_f64m1(vs, __riscv_vfmv_v_f_f64m1(init, gvl), gvl) \
    )
#endif

int CNAME(BLASLONG n, BLASLONG k, FLOAT alpha_r, FLOAT alpha_i,
          FLOAT *a, BLASLONG lda, 
          FLOAT *x, BLASLONG incx, FLOAT *y, BLASLONG incy, void *buffer) 
{
    BLASLONG i;
    unsigned int gvl;
    FLOAT_V_T va_real, va_imag, vx_real, vx_imag;
    FLOAT *X = x;
    FLOAT *Y = y;
    FLOAT *buffer_alloc = (FLOAT *)buffer;

    /* 处理非连续内存访问 */
    if (incy != 1) {
        Y = buffer_alloc;
        buffer_alloc += n * 2;
        COPY_K(n * 2, y, incy, Y, 1);
    }
    if (incx != 1) {
        X = buffer_alloc;
        COPY_K(n * 2, x, incx, X, 1);
    }

#ifndef LOWER
    BLASLONG offset = k;
#endif

    for (i = 0; i < n; i++) {
#ifndef LOWER
        BLASLONG length = k - offset;
        BLASLONG a_idx = i * lda * 2 - offset * 2;
#else
        BLASLONG length = (n - i - 1 < k) ? (n - i - 1) : k;
        BLASLONG a_idx = i * lda * 2 + 2;
#endif

        /* 对角元素处理 */
        FLOAT diag_real = a[i*lda*2];
        FLOAT diag_imag = a[i*lda*2+1];
        FLOAT x_real = X[i*2];
        FLOAT x_imag = X[i*2+1];
        
        Y[i*2]   += alpha_r*(diag_real*x_real - diag_imag*x_imag)
                  - alpha_i*(diag_real*x_imag + diag_imag*x_real);
        Y[i*2+1] += alpha_r*(diag_real*x_imag + diag_imag*x_real)
                  + alpha_i*(diag_real*x_real - diag_imag*x_imag);

        /* 带状区域向量化处理 */
        if (length > 0) {
            BLASLONG x_offset = (i - length)*2;
            BLASLONG remain = length;
            
            while(remain > 0){
                gvl = VSETVL(remain);
                
                /* 加载矩阵元素 */
                va_real = VLSEV_FLOAT(&a[a_idx], lda*2*sizeof(FLOAT), gvl);
                va_imag = VLSEV_FLOAT(&a[a_idx+sizeof(FLOAT)], lda*2*sizeof(FLOAT), gvl);
                
                /* 加载向量元素 */
                vx_real = VLSEV_FLOAT(&X[x_offset], 2*sizeof(FLOAT), gvl);
                vx_imag = VLSEV_FLOAT(&X[x_offset+sizeof(FLOAT)], 2*sizeof(FLOAT), gvl);

                /* 复数乘法计算 */
                FLOAT_V_T vdot_real = VFMVVF_FLOAT(0, gvl);
                FLOAT_V_T vdot_imag = VFMVVF_FLOAT(0, gvl);

#if defined(HEMVREV)
                /* 共轭处理 */
                va_imag = VFNMSACVV_FLOAT(va_imag, VFMVVF_FLOAT(-1, gvl), va_imag, gvl);
#endif
                vdot_real = VFMACCVV_FLOAT(vdot_real, va_real, vx_real, gvl);
                vdot_real = VFNMSACVV_FLOAT(vdot_real, va_imag, vx_imag, gvl);
                vdot_imag = VFMACCVV_FLOAT(vdot_imag, va_real, vx_imag, gvl);
                vdot_imag = VFMACCVV_FLOAT(vdot_imag, va_imag, vx_real, gvl);

                /* 关键修正：正确归约到标量 */
                FLOAT sum_real = VFREDSUM_FLOAT(vdot_real, 0.0f, gvl);
                FLOAT sum_imag = VFREDSUM_FLOAT(vdot_imag, 0.0f, gvl);

                /* 累加结果 */
                Y[i*2]   += alpha_r*sum_real - alpha_i*sum_imag;
                Y[i*2+1] += alpha_r*sum_imag + alpha_i*sum_real;

                /* 更新索引 */
                remain -= gvl;
                a_idx += gvl*lda*2;
                x_offset += gvl*2;
            }
        }

#ifndef LOWER
        if (offset > 0) offset--;
#endif
    }

    /* 写回非连续内存 */
    if (incy != 1) {
        COPY_K(n*2, Y, 1, y, incy);
    }

    return 0;
}