//
// Created by Wassim Omais on 5/3/21.
//

#ifndef FASTFFT_FFT_H
#define FASTFFT_FFT_H

#include <Convolution.h>
#include <ConvolutionSlow.h>
#include <fft/FFTPrecomp.h>
#include <Complex.h>
#include <assert.h>
#include <omp.h>
#include <immintrin.h>

#include <algorithm>

namespace FFTWrapper {
    using std::vector;

    template<typename T, typename U = double, typename Resultant = U>
    class FFT : public Convolution<T, Resultant> {
        static_assert(std::is_floating_point<U>::value, "Computation type must be floating point.");
    private:
        using Complex = complex<U>;

        virtual void dft(vector<Complex> &a, bool invert) const = 0;

        int ncores_;
    public:
        int cores() const { return ncores_; }

        FFT(int cores) : ncores_(cores) {}
        FFT() : FFT(1) {}

        vector<Resultant> convolve(const vector<T> &a, const vector<T> &b) const override {
            vector<Complex> va(a.begin(), a.end());
            vector<Complex> vb(b.begin(), b.end());
            size_t n = 1;
            while (n < va.size() + vb.size())
                n <<= 1;
            if (n <= 16)
                return ConvolutionSlow<T, Resultant>().convolve(a, b);
            va.resize(n), vb.resize(n);
            dft(va, false);
            dft(vb, false);
#pragma omp parallel for num_threads(ncores_)
            for (int i = 0; i < (n >> 2); ++i) {
                /* Let each thread operate on a separate cache line. */
                va[i << 2] *= vb[i << 2];
                va[i << 2 | 1] *= vb[i << 2 | 1];
                va[i << 2 | 2] *= vb[i << 2 | 2];
                va[i << 2 | 3] *= vb[i << 2 | 3];
            }
            dft(va, true);
            vector<Resultant> result(a.size() + b.size() - 1);
#pragma omp parallel for num_threads(ncores_)
            for (int i = 0; i < result.size(); ++i) {
                if constexpr (std::is_integral<Resultant>::value) {
                    // TODO: there are almost certainly precision problems here!
                    result[i] = static_cast<Resultant>(std::llround(va[i].real()));
                } else {
                    result[i] = va[i].real();
                }
            }
            return result;
        }
    }; // FFT<T>

    template<typename T, typename U = double, typename Resultant = U>
    class FFTRecursive : public FFT<T, U, Resultant> {
    private:
        using Complex = complex<U>;

        virtual void merge(vector<Complex> &a,
                           vector<Complex> &even,
                           vector<Complex> &odd,
                           bool invert) const {
            const U PI = std::acos(static_cast<U>(-1.0));
            const int n = a.size();
            U ang = 2 * PI / n * (invert ? -1 : 1);
            Complex w(1), root_of_unity(std::cos(ang), std::sin(ang));
            for (int i = 0; 2 * i < n; ++i) {
                a[i] = even[i] + w * odd[i];
                a[i + n / 2] = even[i] - w * odd[i];
                if (invert) {
                    a[i] /= 2;
                    a[i + n / 2] /= 2;
                }
                w *= root_of_unity;
            }
        }

        void dft(vector<Complex> &a, bool invert) const override {
            const int n = a.size();
            if (n == 1)
                return;
            // n must be power of two!
            assert(n == (n & -n));
            vector<Complex> even(n / 2), odd(n / 2);
            for (int i = 0; 2 * i < n; ++i) {
                even[i] = a[2 * i];
                odd[i] = a[2 * i + 1];
            }
            dft(even, invert);
            dft(odd, invert);
            merge(a, even, odd, invert);
        }

    public:
        const char *name() const override { return "Sequential Recursive FFT"; }
    }; // FFTRecursive

    template<typename T, typename U = double, typename Resultant = U>
    class FFTIterative : public FFT<T, U, Resultant> {
    public:
        const char *name() const override { return "Iterative In-Place Sequential FFT."; }

        FFTIterative(int cores) : FFT<T, U, Resultant>(cores) {}

        FFTIterative() : FFT<T, U, Resultant>() {}

    private:
        using Complex = complex<U>;

        void dft(vector<Complex> &a, bool invert) const override {
            int n = a.size();
            /* for AVX-512 loading... */
            std::array<double, 8> just_n;
            just_n.fill((double) n);

            int lg = 31 - __builtin_clz(n);
            int lg_cap = 31 - __builtin_clz(FFTPrecomp<U>::capacity);
            int shift = lg_cap - lg;
            const int cores = FFT<T, U, Resultant>::cores();
#pragma omp parallel for num_threads(cores) schedule(dynamic)
            for (int i = 0; i < (n >> 4); i++) {
                for (int k = 0; k < (1 << 4); ++k) {
                    if ((i << 4 | k) < (FFTPrecomp<U>::rev[i << 4 | k] >> shift)) {
                        std::swap(a[i << 4 | k], a[FFTPrecomp<U>::rev[i << 4 | k] >> shift]);
                    }
                }
            }
            for (int lg_len = 1, len = 2; len <= n; len <<= 1, lg_len += 1) {
                const int num_half_intervals = n >> lg_len;
                int blocks_per_half = std::min(len >> (4 + 1),
                                               2 * (cores + num_half_intervals - 1) / num_half_intervals);
                if (blocks_per_half == 0) blocks_per_half = len >> 1;
                const int block_size = ((len >> 1) + blocks_per_half - 1) / blocks_per_half;
                const int num_blocks = blocks_per_half * num_half_intervals;
#pragma omp parallel for num_threads(cores)
                for (int blk = 0; blk < num_blocks; ++blk) {
                    const int which_half = blk / blocks_per_half;
                    const int block_ind = blk % blocks_per_half;
                    const int half_start = (which_half << lg_len);
                    const int start = half_start + block_ind * block_size;
                    const int end = std::min(half_start + (len >> 1), start + block_size);
                    for (int j = 0; j < end - start; ++j) {
                        int ind = start + j - half_start;
                        Complex w = FFTPrecomp<U>::roots[len + ind];
                        w.dat[1] *= 1 - (static_cast<int>(invert) << 1);
                        Complex u = a[start + j], v = a[start + j + (len >> 1)] * w;
                        a[start + j] = u + v;
                        a[start + j + (len >> 1)] = u - v;
                    }
                }
            }
            if (invert) {
#pragma omp parallel for num_threads(cores)
                for (int i = 0; i < (n >> 2); ++i) {
                    if constexpr (sizeof(U) == 8) {
                        __m512d vec = _mm512_load_pd((void *) &a[i << 2]);
                        __m512d quot = _mm512_load_pd((void *)just_n.data());
                        __m512d ree =  _mm512_div_pd(vec, quot);
                        _mm512_store_pd((void *)&a[i << 2], ree);
                    } else {
                        a[i << 2] /= n;
                        a[i << 2 | 1] /= n;
                        a[i << 2 | 2] /= n;
                        a[i << 2 | 3] /= n;
                    }
                }
            }
        }
    }; // FFTIterative
} // namespace FFTWrapper

using FFTWrapper::FFT;
using FFTWrapper::FFTRecursive;
using FFTWrapper::FFTIterative;

#endif //FASTFFT_FFT_H
