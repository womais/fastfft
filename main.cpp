#include <iostream>
#include <memory>
#include <iomanip>
#include <thread>
#include <fft/FFT.h>
#include <ConvolutionSlow.h>

using std::vector;

void run_convolve_tests(int n, int T) {
    srand(time(0));
    const auto processor_count = std::thread::hardware_concurrency();
    for (int ncores = 1; ncores <= processor_count; ++ncores) {
        FFTIterative<int, long double, long long, 1 << 20> fast_fft(ncores);
        for (int i = 0; i < T; ++i) {
            vector<int> a(n), b(n);
            for (int j = 0; j < n; ++j) {
                a[j] = rand() % (int) 1e6;
                b[j] = rand() % (int) 1e6;
            }
            vector<long long> axb_one, axb_two;
            std::cout << "Fast Parallel FFT, " << ncores << " core(s): " << fast_fft.timed_convolve(a, b, axb_one) << "\n";
        }
    }
}

int main() {
    std::cout << "How many tests to run?\n";
    int T;
    std::cin >> T;
    run_convolve_tests(100'000, T);
    return 0;
}
