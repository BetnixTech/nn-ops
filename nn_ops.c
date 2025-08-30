#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(__AVX__)
#include <immintrin.h>
#endif

// =======================================================
// Matrix Multiply (SIMD accelerated if available)
// =======================================================
void matmul(const float* A, const float* B, float* C, int M, int K, int N) {
#if defined(__AVX__)
    // AVX version (process 8 floats at a time)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            __m256 sum_vec = _mm256_setzero_ps();
            int k;
            for (k = 0; k + 8 <= K; k += 8) {
                __m256 a_vec = _mm256_loadu_ps(&A[i*K + k]);
                __m256 b_vec = _mm256_set_ps(
                    B[(k+7)*N + j], B[(k+6)*N + j],
                    B[(k+5)*N + j], B[(k+4)*N + j],
                    B[(k+3)*N + j], B[(k+2)*N + j],
                    B[(k+1)*N + j], B[(k+0)*N + j]
                );
                sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
            }
            float sum[8];
            _mm256_storeu_ps(sum, sum_vec);
            float total = sum[0]+sum[1]+sum[2]+sum[3]+sum[4]+sum[5]+sum[6]+sum[7];
            for (; k < K; k++) total += A[i*K+k] * B[k*N+j];
            C[i*N + j] = total;
        }
    }
#elif defined(__ARM_NEON)
    // NEON version (process 4 floats at a time)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float32x4_t sum_vec = vdupq_n_f32(0.0f);
            int k;
            for (k = 0; k + 4 <= K; k += 4) {
                float32x4_t a_vec = vld1q_f32(&A[i*K + k]);
                float32x4_t b_vec = {B[(k+0)*N + j], B[(k+1)*N + j],
                                     B[(k+2)*N + j], B[(k+3)*N + j]};
                sum_vec = vmlaq_f32(sum_vec, a_vec, b_vec);
            }
            float32x2_t sum_pair = vadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
            float total = vget_lane_f32(sum_pair, 0) + vget_lane_f32(sum_pair, 1);
            for (; k < K; k++) total += A[i*K+k] * B[k*N+j];
            C[i*N + j] = total;
        }
    }
#else
    // Portable fallback
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i*K+k] * B[k*N+j];
            }
            C[i*N + j] = sum;
        }
    }
#endif
}

// =======================================================
// Activations
// =======================================================
void relu(float* X, int size) {
    for (int i = 0; i < size; i++) {
        if (X[i] < 0.0f) X[i] = 0.0f;
    }
}

void softmax(float* X, int M, int N) {
    for (int i = 0; i < M; i++) {
        float max_val = -FLT_MAX;
        for (int j = 0; j < N; j++)
            if (X[i*N + j] > max_val) max_val = X[i*N + j];

        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            X[i*N + j] = expf(X[i*N + j] - max_val);
            sum += X[i*N + j];
        }
        for (int j = 0; j < N; j++)
            X[i*N + j] /= sum;
    }
}

// =======================================================
// Conv2D (NCHW layout)
// =======================================================
void conv2d(const float* input, const float* kernel, const float* bias,
            float* output, int batch, int in_c, int in_h, int in_w,
            int out_c, int k_h, int k_w, int stride, int pad) {

    int out_h = (in_h - k_h + 2*pad) / stride + 1;
    int out_w = (in_w - k_w + 2*pad) / stride + 1;

    for (int b = 0; b < batch; b++) {
        for (int oc = 0; oc < out_c; oc++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    float sum = bias ? bias[oc] : 0.0f;
                    for (int ic = 0; ic < in_c; ic++) {
                        for (int kh = 0; kh < k_h; kh++) {
                            for (int kw = 0; kw < k_w; kw++) {
                                int ih = oh*stride + kh - pad;
                                int iw = ow*stride + kw - pad;
                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    int in_idx  = ((b*in_c + ic)*in_h + ih)*in_w + iw;
                                    int ker_idx = ((oc*in_c + ic)*k_h + kh)*k_w + kw;
                                    sum += input[in_idx] * kernel[ker_idx];
                                }
                            }
                        }
                    }
                    int out_idx = ((b*out_c + oc)*out_h + oh)*out_w + ow;
                    output[out_idx] = sum;
                }
            }
        }
    }
}

// =======================================================
// Example
// =======================================================
#ifdef TEST_NN_OPS
int main() {
    // Dense layer example
    int batch = 1, in_dim = 4, out_dim = 3;
    float X[4] = {1, 2, 3, 4};
    float W[12] = {0.1f,0.2f,0.3f,
                   0.4f,0.5f,0.6f,
                   0.7f,0.8f,0.9f,
                   1.0f,1.1f,1.2f};
    float b[3] = {0.01f,0.02f,0.03f};
    float Y[3];
    matmul(X, W, Y, batch, in_dim, out_dim);
    for (int i=0;i<out_dim;i++) Y[i]+=b[i];
    relu(Y, out_dim);
    printf("Dense+ReLU: ");
    for (int i=0;i<out_dim;i++) printf("%f ", Y[i]);
    printf("\n");

    // Conv2D example: 1x1x3x3 input, 1 filter 3x3
    float input[9] = {1,2,3,4,5,6,7,8,9};
    float kernel[9] = {1,0,-1,1,0,-1,1,0,-1};
    float bias_c[1] = {0.0f};
    float out[1];
    conv2d(input, kernel, bias_c, out, 1, 1, 3, 3, 1, 3, 3, 1, 0);
    printf("Conv2D result: %f\n", out[0]);
    return 0;
}
#endif
