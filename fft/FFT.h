//
// Created by Wassim Omais on 5/3/21.
//

#ifndef FASTFFT_FFT_H
#define FASTFFT_FFT_H

#include <Convolution.h>
#include <fft/FFTPrecomp.h>
#include <assert.h>
#include <omp.h>

#include <algorithm>
#include <complex>
#include <string>
#include <cmath>

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
        static void prepare_to(int w, int cores) { FFTPrecomp<U>::initialize_to(w, cores); }

        FFT(int cores) : ncores_(cores) {}
        FFT() : FFT(1) {}

        vector<Resultant> convolve(const vector<T> &a, const vector<T> &b) const override {
            vector<Complex> va(a.begin(), a.end());
            vector<Complex> vb(b.begin(), b.end());
            size_t n = 1;
            while (n < va.size() + vb.size())
                n <<= 1;
            va.resize(n), vb.resize(n);
            dft(va, false);
            dft(vb, false);
#pragma omp parallel for num_threads(ncores_)
            for (int i = 0; i < n; ++i) { 
                va[i] *= vb[i];
            }
            dft(va, true);
            vector<Resultant> result(n);
#pragma omp parallel for num_threads(ncores_)
            for (int i = 0; i < n; ++i) {
                if constexpr (std::is_arithmetic<Resultant>::value) {
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
            int lg = 31 - __builtin_clz(n);
            int lg_cap = 31 - __builtin_clz(FFTPrecomp<U>::capacity);
            int shift = lg_cap - lg;
            const int cores = FFT<T, U, Resultant>::cores();
#pragma omp parallel for num_threads(cores) schedule(dynamic)
            for (int i = 0; i < n; i++) {
                if (i < (FFTPrecomp<U>::rev[i] >> shift))  {
                    std::swap(a[i], a[FFTPrecomp<U>::rev[i] >> shift]);
                }
            }
            for (int len = 2; len <= n; len <<= 1) {
                const int num_half_intervals = n >> lg;
                const int blocks_per_half = std::min(len >> 1, 2 * (cores + num_half_intervals - 1) / num_half_intervals);
                const int block_size = ((len >> 1) + blocks_per_half - 1) / blocks_per_half;
                const int num_blocks = blocks_per_half * num_half_intervals;
#pragma omp parallel for num_threads(cores)
                for (int blk = 0; blk < num_blocks; ++blk) {
                    const int which_half = blk / blocks_per_half;
                    const int block_ind = blk % blocks_per_half;
                    const int half_start = (which_half << lg);
                    const int start = half_start + block_ind * block_size;
                    const int end = std::min(half_start + (len >> 1), start + block_size);
                    for (int j = 0; j < end - start; ++j) {
                        int ind = start + j - half_start;
                        if (invert) ind = len - ind;
                        ind = ind & (len - 1);
                        Complex w = FFTPrecomp<U>::roots[len + ind];
                        Complex u = a[start + j], v = a[start + j + len / 2] * w;
                        a[start + j] = u + v;
                        a[start + j + len / 2] = u - v;
                    }
                }
            }
            if (invert) {
#pragma omp parallel for num_threads(cores)
                for (int i = 0; i < n; ++i) {
                    a[i] /= n;
                }
            }
        }
    }; // FFTIterative
} // namespace FFTWrapper

using FFTWrapper::FFT;
using FFTWrapper::FFTRecursive;
using FFTWrapper::FFTIterative;

#endif //FASTFFT_FFT_H
