//
// Created by Wassim Omais on 5/3/21.
//

#ifndef FASTFFT_FFT_H
#define FASTFFT_FFT_H

#include <Convolution.h>
#include <fft/RootsOfUnity.h>
#include <assert.h>
#include <omp.h>

#include <complex>
#include <string>
#include <cmath>

namespace FFTWrapper {
    using std::vector;

    template<typename T, typename U = double, typename Resultant = U, int CAP = 0>
    class FFT : public Convolution<T, Resultant> {
        static_assert(CAP == 0 || (CAP > 0 && CAP == (CAP & -CAP)), "If specified, capacity must be power of two.");
        static_assert(std::is_floating_point<U>::value, "Computation type must be floating point.");
    private:
        using Complex = std::complex<U>;

        virtual void dft(vector<Complex> &a, bool invert) const = 0;

        int ncores_;
    public:
        int cores() const { return ncores_; }

        FFT(int cores) : ncores_(cores) {
            if constexpr (CAP > 0) {
                RootsOfUnity<U>::initialize_to(CAP);
            }
        }

        FFT() : FFT(1) {}

        vector<Resultant> convolve(const vector<T> &a, const vector<T> &b) const override {
            vector<Complex> va(a.begin(), a.end());
            vector<Complex> vb(b.begin(), b.end());
            size_t n = 1;
            while (n < va.size() + vb.size())
                n <<= 1;
            assert(CAP == 0 || CAP >= n);
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

    template<typename T, typename U = double, typename Resultant = U, int CAP = 0>
    class FFTRecursive : public FFT<T, U, Resultant, CAP> {
    private:
        using Complex = std::complex<U>;

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

    template<typename T, typename U = double, typename Resultant = U, int CAP = 0>
    class FFTIterative : public FFT<T, U, Resultant, CAP> {
    public:
        const char *name() const override { return "Iterative In-Place Sequential FFT."; }

        FFTIterative(int cores) : FFT<T, U, Resultant, CAP>(cores) {}

        FFTIterative() : FFT<T, U, Resultant, CAP>() {}

    private:
        using Complex = std::complex<U>;

        void dft(vector<Complex> &a, bool invert) const override {
            const U PI = acos(static_cast<U>(-1));
            int n = a.size();
            for (int i = 1, j = 0; i < n; ++i) {
                int bit = n >> 1;
                for (; j & bit; bit >>= 1)
                    j ^= bit;
                j ^= bit;
                if (i < j) {
                    swap(a[i], a[j]);
                }
            }
            for (int len = 2; len <= n; len <<= 1) {
#pragma omp parallel for num_threads(FFT<T, U, Resultant, CAP>::cores()) schedule(dynamic)
                for (int i = 0; i < n; i += len) {
                    for (int j = 0; j < len / 2; ++j) {
                        Complex w;
                        // if we specified precomputation in our template, be sure to use that.
                        if constexpr (CAP > 0) {
                            int ind = (invert ? len - j : j);
                            w = RootsOfUnity<U>::roots[31 - __builtin_clz(len)][ind];
                        } else {
                            U ang = 2 * PI * j / len * (invert ? -1 : 1);
                            w = Complex(cos(ang), sin(ang));
                        }
                        Complex u = a[i + j], v = a[i + j + len / 2] * w;
                        a[i + j] = u + v;
                        a[i + j + len / 2] = u - v;
                    }
                }
            }
            if (invert) {
                for (Complex &x : a) {
                    x /= n;
                }
            }
        }
    }; // FFTIterative
} // namespace FFTWrapper

using FFTWrapper::FFT;
using FFTWrapper::FFTRecursive;
using FFTWrapper::FFTIterative;

#endif //FASTFFT_FFT_H
