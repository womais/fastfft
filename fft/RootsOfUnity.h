//
// Created by Wassim Omais on 5/4/21.
//

#ifndef FASTFFT_ROOTSOFUNITY_H
#define FASTFFT_ROOTSOFUNITY_H

#include <assert.h>
#include <array>
#include <complex>
#include <cmath>

// Precomputation of exp(2 * i * k * pi / SIZE).
// Ideally could do some stuff at compile time upon request,
// but sin() and cos() can't be constexpr without a gcc extension.
template <typename U>
struct RootsOfUnity {
    using Complex = std::complex<U>;
    static std::vector<Complex> roots[32];
    static bool initialized[32];
    static void initialize_to(int w) {
        assert(w > 0 && w == (w & -w));
        static const U PI = std::acos(static_cast<U>(-1));
        for (int i = 1; i <= w; i <<= 1) {
            int bit = 31 - __builtin_clz(i);
            if (initialized[bit]) continue;
            roots[bit].resize(i + 1);
            for (int j = 0; j <= i; ++j)
                roots[bit][j] = Complex(std::cos(2 * PI * j / i), std::sin(2 * PI * j / i));
            initialized[bit] = true;
        }
    }
};

template <typename U>
bool RootsOfUnity<U>::initialized[32];

template <typename U>
std::vector<std::complex<U>> RootsOfUnity<U>::roots[32];

#endif //FASTFFT_ROOTSOFUNITY_H
