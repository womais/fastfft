#ifndef FASTFFT_CONVOLUTION_H
#define FASTFFT_CONVOLUTION_H

#include <vector>
#include <string>
#include <chrono>
#include <aligned_allocator.h>

namespace ConvolutionWrapper {
    using std::vector;

    template<typename T, typename Allocator = aligned_allocator<T, 64>>
    class Convolution {
    public:
        virtual void convolve(vector<T, Allocator>& a, vector<T, Allocator>& b, int keep) const = 0;

        /* Required size for both input vectors to a convolution.
         * By default, it's just the sum minus one (large enough to hold the conv.)
         * but it might be larger. For example, FFT requires a larger one.
         */
        virtual int required_size(int x, int y) const { return x + y - 1; }

        /* The name of the convolution (for printing purposes). */
        virtual const char *name() const = 0;

        /* A wrapper that calls convolve with a timer.
         * The convolution is done in-place, and data in a is overwritten.
         * At the end, a.size() == keep and it contains the multiplication.
         * Original data is purge (copy it initially if you want).
         * It was decided to not copy inside the convolutions, because
         * that was unnecessary time wasted (and not insignificant, either).
         */
        double timed_convolve(vector<T, Allocator>& a,
                              vector<T, Allocator>& b,
                              int keep) const {
            // return value in ms
            auto start = std::chrono::high_resolution_clock::now();
            this->convolve(a, b, keep);
            auto end = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(end - start).count();
        }
    }; // Convolution
} // namespace ConvolutionWrapper

using ConvolutionWrapper::Convolution;

#endif //FASTFFT_CONVOLUTION_H
