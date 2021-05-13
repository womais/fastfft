#ifndef FASTFFT_CONVOLUTIONSLOW_H
#define FASTFFT_CONVOLUTIONSLOW_H

#include <Convolution.h>

template<typename T>
class ConvolutionSlow : public Convolution<T> {
public:
    const char *name() const override { return "Slow convolution."; }
    void convolve(std::vector<T, aligned_allocator<T, 64>>& a,
                  std::vector<T, aligned_allocator<T, 64>>& b,
                  int keep) const override {
        std::vector<T, aligned_allocator<T, 64> > result(keep);
        for (int i = 0; i < keep; ++i) {
            for (int j = 0; j < keep - i; ++j) {
                result[i + j] += a[i] * b[j];
            }
        }
        a = std::move(result);
    }
};

#endif //FASTFFT_CONVOLUTIONSLOW_H
