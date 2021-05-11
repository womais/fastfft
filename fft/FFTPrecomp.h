//
// Created by Wassim Omais on 5/4/21.
//

#ifndef FASTFFT_FFTPRECOMP_H
#define FASTFFT_FFTPRECOMP_H

#include <Complex.h>
#include <aligned_allocator.h>
#include <assert.h>
#include <array>
#include <cmath>

// Precomputation of exp(2 * i * k * pi / SIZE).
// Ideally could do some stuff at compile time upon request,
// but sin() and cos() can't be constexpr without a gcc extension.
template<typename U>
struct FFTPrecomp {
    using Complex = complex<U>;
    using vec_cmplx = std::vector<Complex, aligned_allocator<Complex, 512> >;
    static vec_cmplx roots;
    static std::vector<int> rev;
    static int capacity;
    static void initialize_to(int w, int cores = 1) {
        if (w <= capacity)
            return;
        assert(w > 0 && w == (w & -w));
        capacity = w;
        rev.resize(w);
        roots.resize(w << 1);
        int lg = 31 - __builtin_clz(w);
//#pragma omp parallel for num_threads(cores)
        for (int i = 0; i < w; i++)
            rev[i] = (rev[i >> 1] >> 1) + ((i & 1) << (lg - 1));
        static const U PI = std::acos(static_cast<U>(-1));
#pragma omp parallel for num_threads(cores) schedule(dynamic)
        for (int i = 1; i <= lg; ++i) {
            int base = 1 << i;
            for (int k = 0; k < base; ++k) {
                double ang = 2 * PI * k / base;
                roots[base + k].dat[0] = std::cos(ang);
                roots[base + k].dat[1] = std::sin(ang);
            }
        }
    }
};
template <typename U>
int FFTPrecomp<U>::capacity = 0;

template <typename U>
typename FFTPrecomp<U>::vec_cmplx FFTPrecomp<U>::roots;

template <typename U>
std::vector<int> FFTPrecomp<U>::rev;

#endif //FASTFFT_FFTPRECOMP_H
