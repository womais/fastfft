//
// Created by Wassim Omais on 5/4/21.
//

#ifndef FASTFFT_CONVOLUTIONSLOW_H
#define FASTFFT_CONVOLUTIONSLOW_H

#include <Convolution.h>

template<typename T, typename U>
class ConvolutionSlow : public Convolution<T, U> {
public:
    const char *name() const override { return "Slow convolution."; }
    std::vector<U> convolve(const std::vector<T>& a, const std::vector<T>& b) const override {
        std::vector<U> result(a.size() + b.size());
        for (int i = 0; i < a.size(); ++i) {
            for (int j = 0; j < b.size(); ++j) {
                result[i + j] += static_cast<U>(a[i]) * b[j];
            }
        }
        return result;
    }
};

#endif //FASTFFT_CONVOLUTIONSLOW_H
