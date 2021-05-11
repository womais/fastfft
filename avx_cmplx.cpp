#include <avx_cmplx.h>

__m256d avx_cmplx_mul(__m256d v, __m256d w)
{
    auto neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
    auto prod = _mm256_mul_pd(v, w);
    w = _mm256_permute_pd(w, 0x5);
    w = _mm256_mul_pd(w, neg);
    auto prod2 = _mm256_mul_pd(v, w);
    v = _mm256_hsub_pd(prod, prod2);
    return v;
}
