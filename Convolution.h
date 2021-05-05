//
// Created by Wassim Omais on 5/3/21.
//

#ifndef FASTFFT_CONVOLUTION_H
#define FASTFFT_CONVOLUTION_H

#include <vector>
#include <string>
#include <chrono>

namespace ConvolutionWrapper {
    using std::vector;

    template<typename T, typename Resultant>
    class Convolution {
        static_assert(std::is_arithmetic<Resultant>::value, "Resultant type must be arithmetic type.");
        static_assert(std::is_arithmetic<T>::value, "Input type must be arithmetic type.");
    public:
        virtual vector<Resultant> convolve(const vector<T> &a, const vector<T> &b) const = 0;
        virtual const char *name() const = 0;
        double timed_convolve(const vector<T>& a, const vector<T>& b, vector<Resultant>& r) {
            // return value in ms
            auto start = std::chrono::high_resolution_clock::now();
            r = this->convolve(a, b);
            auto end = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(end - start).count();
        }
    }; // Convolution
} // namespace ConvolutionWrapper

using ConvolutionWrapper::Convolution;

#endif //FASTFFT_CONVOLUTION_H
