#include <iostream>
#include <memory>
#include <iomanip>
#include <thread>
#include <fft/FFT.h>
#include <ConvolutionSlow.h>
#include <numeric>

using std::vector;

void run_convolve_tests(int n, int T) {
    srand(time(0));
    vector<int> a(n), b(n);
    for (int i = 0; i < n; ++i) {
        a[i] = rand() % (int) 1e6;
        b[i] = rand() % (int) 1e6;
    }
    const auto processor_count = std::thread::hardware_concurrency();
    FFTPrecomp<long double>::initialize_to(1 << 15, 1);
    for (int ncores = 1; ncores <= processor_count - 1; ++ncores) {
        FFTIterative<int, long double, long long> fast_fft(ncores);
        ConvolutionSlow<int, long long> slow_conv;
        for (int i = 0; i < T; ++i) {
            vector<long long> axb_two;
            vector<long long> axb_one;
            std::cout << "Fast (" << ncores << "): " << fast_fft.timed_convolve(a, b, axb_two) << "\n";
            std::cout << "Slow: " << slow_conv.timed_convolve(a, b, axb_one) << "\n";
            for (int j = 0; j < 2 * n; ++j) {
                assert(axb_one[j] == axb_two[j]);
            }
        }
    }
}

int main() {
    std::cout << "How many tests to run?\n";
    int T;
    std::cin >> T;
    run_convolve_tests(1000, T);
    return 0;
}
