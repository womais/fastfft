#ifndef FASTFFT_AVX_CMPLX_H
#define FASTFFT_AVX_CMPLX_H

#include <immintrin.h>

// Multiplies vectors in v and w as complex numbers.
// Each vector stores 4 complex numbers.
// v[2 * i] is the real part of complex number i in vector v (starting from i = 0)
// v[2 * i + 1] is the imaginary part.
__m256d avx_cmplx_mul(__m256d v, __m256d w);

#endif

