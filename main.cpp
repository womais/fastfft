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
    std::iota(a.begin(), a.end(), 0);
    std::iota(b.begin(), b.end(), 0);
    const auto processor_count = std::thread::hardware_concurrency();
    for (int ncores = 1; ncores < processor_count; ++ncores) {
        FFTIterative<int, long double, long long, 1 << 21> fast_fft(ncores);
        ConvolutionSlow<int, long long> shit;
        for (int i = 0; i < T; ++i) {
            vector<long long> axb_two;
            std::cout << "Fast (" << ncores << "): " << fast_fft.timed_convolve(a, b, axb_two) << "\n";
        }
    }
}

int main() {
    std::cout << "How many tests to run?\n";
    int T;
    std::cin >> T;
    run_convolve_tests(1'000'000, T);
    return 0;
}
