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
        a[i] = rand() % (int) 1000;
        b[i] = rand() % (int) 1000;
    }
    const auto processor_count = std::thread::hardware_concurrency();
    FFTPrecomp<double>::initialize_to(1 << 25, 1);
    for (int ncores = 1; ncores <= processor_count - 1; ++ncores) {
        FFTIterative<int, double, long long> fast_fft(ncores);
        vector<long long> out;
        std::cout << "Time (" << ncores << "): " << fast_fft.timed_convolve(a, b, out) << "\n";
    }
}

int main() {
    std::cout << "How many tests to run?\n";
    int T;
    std::cin >> T;
    run_convolve_tests(10'000'000, T);
    return 0;
}
