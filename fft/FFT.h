#ifndef FASTFFT_FFT_H
#define FASTFFT_FFT_H

#include <Convolution.h>
#include <ConvolutionSlow.h>
#include <fft/FFTPrecomp.h>
#include <Complex.h>
#include <assert.h>
#include <omp.h>
#include <immintrin.h>
#include <avx_cmplx.h>
#include <algorithm>
#include <fft/gpu_tools.h>

namespace FFTWrapper {
    using std::vector;
    // Currently, only U = double is supported... 
    template<typename U = double>
    class FFT : public Convolution<complex<U>> {
        static_assert(std::is_floating_point<U>::value, "Computation type must be floating point.");
    public:
        using vec_cmplx = vector<complex<U>, aligned_allocator<complex<U>, 64>>;
        int required_size(int x, int y) const override {
            int k = 1;
            while (k < x + y)
                k <<= 1;
            return k;
        }
    private:
        using Complex = complex<U>;
        virtual void dft(vec_cmplx& a) const = 0;
        virtual void dft_inverse(vec_cmplx& a) const = 0;
        int ncores_;
    public:
        int cores() const { return ncores_; }

        FFT(int cores) : ncores_(cores) {}
        FFT() : FFT(1) {}

        void convolve(vec_cmplx& va, vec_cmplx& vb, int keep) const override {
            if (keep <= 16)
                return ConvolutionSlow<Complex>().convolve(va, vb, keep);
            const int n = va.size();
            va.resize(n), vb.resize(n);
            dft(va); dft(vb);
#pragma omp parallel for num_threads(ncores_)
            for (int i = 0; i < (n >> 2); ++i) {
                /* Let each thread operate on a separate cache line. */
                va[i << 2] *= vb[i << 2];
                va[i << 2 | 1] *= vb[i << 2 | 1];
                va[i << 2 | 2] *= vb[i << 2 | 2];
                va[i << 2 | 3] *= vb[i << 2 | 3];
            }
            dft_inverse(va);
            va.resize(keep);
        }
    }; // FFT<T>

    template<typename U = double>
    class FFTRecursive : public FFT<U> {
    private:
        using Complex = complex<U>;
        using vec_cmplx = vector<Complex, aligned_allocator<Complex, 64> >;

        virtual void merge(vec_cmplx &a,
                           vec_cmplx &even,
                           vec_cmplx &odd,
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
        template <bool invert>
        void dft_(vec_cmplx &a) const {
            const int n = a.size();
            if (n == 1)
                return;
            // n must be power of two!
            assert(n == (n & -n));
            vec_cmplx even(n / 2), odd(n / 2);
            for (int i = 0; 2 * i < n; ++i) {
                even[i] = a[2 * i];
                odd[i] = a[2 * i + 1];
            }
            dft_<invert>(even);
            dft_<invert>(odd);
            merge(a, even, odd, invert);
        }

        void dft(vec_cmplx& a) const override { dft_<false>(a); } 
        void dft_inverse(vec_cmplx& a) const override { dft_<true>(a); }

    public:
        const char *name() const override { return "Recursive FFT"; }
    }; // FFTRecursive

    template<typename U = double>
    class FFTIterative : public FFT<U> {
    public:
        const char *name() const override { return "Iterative In-Place FFT."; }

        FFTIterative(int cores) : FFT<U>(cores) {}

        FFTIterative() : FFT<U>() {}

    private:
        using Complex = complex<U>;
        using vec_cmplx = vector<Complex, aligned_allocator<Complex, 64> >;
        template <bool invert>
        void do_pass_cpu(vec_cmplx& a, int len, int lg_len, int n) const {
            const int cores = FFT<U>::cores();
            const int num_half_intervals = n >> lg_len;
            const int blocks_per_half = (len >> 2); 
            const int num_blocks = blocks_per_half * num_half_intervals;
#pragma omp parallel for num_threads(cores)
            for (int blk = 0; blk < num_blocks; ++blk) {
                const int which_half = blk / blocks_per_half;
                const int block_ind = blk % blocks_per_half;
                const int half_start = (which_half << lg_len);
                const int start = half_start + block_ind * 2;
                 __m256d vecw = _mm256_load_pd(reinterpret_cast<double*>(&FFTPrecomp<U>::roots[len + start - half_start]));
                __m256d vecu = _mm256_load_pd(reinterpret_cast<double*>(&a[start]));
                __m256d vecv = _mm256_load_pd(reinterpret_cast<double*>(&a[start + (len >> 1)]));
                if constexpr (invert) {
                    __m256d scale = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
                    vecw = _mm256_mul_pd(vecw, scale);
                }
                vecv = avx_cmplx_mul(vecv, vecw);
                _mm256_store_pd(reinterpret_cast<double*>(&a[start]), _mm256_add_pd(vecu, vecv));
                _mm256_store_pd(reinterpret_cast<double*>(&a[start + (len >> 1)]), _mm256_sub_pd(vecu, vecv));
            }
        }
        template <bool invert>
        void do_pass_gpu(int len, int lg_len, int n) const {
            if constexpr (invert) run_gpu_pass_inv(len, lg_len, n);
            else run_gpu_pass(len, lg_len, n);
        }
        template <bool invert>
        void dft_(vec_cmplx &a) const {
            int n = a.size();
            int lg = 31 - __builtin_clz(n);
            int lg_cap = 31 - __builtin_clz(FFTPrecomp<U>::capacity);
            int shift = lg_cap - lg;
            const int cores = FFT<U>::cores();
#pragma omp parallel for num_threads(cores) schedule(dynamic)
            for (int i = 0; i < (n >> 4); i++) {
                for (int k = 0; k < (1 << 4); ++k) {
                    if ((i << 4 | k) < (FFTPrecomp<U>::rev[i << 4 | k] >> shift)) {
                        std::swap(a[i << 4 | k], a[FFTPrecomp<U>::rev[i << 4 | k] >> shift]);
                    }
                }
            }
            const Complex _w2 = [&]() { 
                if (invert) return Complex(FFTPrecomp<U>::roots[2].dat[0], -FFTPrecomp<U>::roots[2].dat[1]);
                else return FFTPrecomp<U>::roots[2];
            }();
#pragma omp parallel for num_threads(cores)
            for (int i = 0; i < n; i += 2) {
                Complex u = a[i], v = a[i + 1] * _w2;
                a[i] = u + v;
                a[i + 1] = u - v;
            }
            initialize_gpu_data(n, a.data());
            for (int lg_len = 2, len = 4; len <= n; len <<= 1, lg_len += 1)
                do_pass_gpu<invert>(len, lg_len, n);
            finish_gpu_data(n);
            if constexpr (invert) {
#pragma omp parallel for num_threads(cores)
                for (int i = 0; i < (n >> 2); ++i) {
                    a[i << 2] /= n;
                    a[i << 2 | 1] /= n;
                    a[i << 2 | 2] /= n;
                    a[i << 2 | 3] /= n;
                }
            }
        }
        
        void dft(vec_cmplx& a) const override { dft_<false>(a); } 
        void dft_inverse(vec_cmplx& a) const override { dft_<true>(a); }

    }; // FFTIterative
} // namespace FFTWrapper

using FFTWrapper::FFT;
using FFTWrapper::FFTRecursive;
using FFTWrapper::FFTIterative;

#endif //FASTFFT_FFT_H
