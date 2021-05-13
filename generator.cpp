#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <ConvolutionSlow.h>
#include <fft/FFT.h>
#include <random>

int main(int argc, const char* argv[]) {
    if (argc < 3 || argc > 4) {
        std::cout << "Invalid usage (wrong number of arguments).\n";
        exit(1);
    }
    std::random_device device;
    std::mt19937 gen(device());
    std::uniform_int_distribution<int> dist_size(10'000, 100'000);
    std::uniform_int_distribution<int> dist_values(1000, 10'000);

    // So that the result fits in a 64-bit integer, the values for the -large
    // set of test cases will be smaller.
    std::uniform_int_distribution<int> dist_values_large(1, 100);

    std::string path = argv[1];
    int num_tests = atoi(argv[2]);
    ConvolutionSlow<long long> conv_reliable;
    FFTIterative<double> conv_fast(2);
    auto dump_vec = [&](std::ofstream& os, const auto& vec, int terms) {
        using T = typename std::remove_reference<decltype(vec)>::type;
        for (int i = 0; i < terms; ++i) {
            if constexpr (std::is_same<complex<double>, typename T::value_type>::value) {
                os << llround(vec[i].real()) << " ";
            } else {
                os << vec[i] << " ";
            }
        }

        os << "\n";
    };
    for (int i = 1; i <= num_tests; ++i)
    {
        std::cout << "Generating test " << i << "..." << "\n";
        std::ofstream os(path + "/random" + std::to_string(i) + ".ffttest");
        const int size_a = dist_size(gen), size_b = dist_size(gen);
        std::vector<long long, aligned_allocator<long long, 64> >  a(conv_reliable.required_size(size_a, size_b));
        std::vector<long long, aligned_allocator<long long, 64> >  b(conv_reliable.required_size(size_a, size_b));
        for (int j = 0; j < size_a; ++j) a[j] = dist_values(gen);
        for (int j = 0; j < size_b; ++j) b[j] = dist_values(gen);
        os << size_a << " " << size_b << "\n";
        dump_vec(os, a, size_a);
        dump_vec(os, b, size_b);
        conv_reliable.convolve(a, b, size_a + size_b - 1);
        dump_vec(os, a, size_a + size_b - 1);
        std::cout << "OK done!\n\n";
    }
    if (argc == 4 && !strcmp(argv[3], "--include-large")) {
        std::cout << "Preparing for large case generation. Initializing fast FFT...\n";
        FFTPrecomp<double>::initialize_to(1 << 28, 4);
        int size = 1 << 20;
        for (int i = num_tests + 1; i <= num_tests + 7; ++i) {
            std::cout << "Generating large test " << i - num_tests << " ..." << "\n";
            std::ofstream os(path + "/random-large" + std::to_string(i - num_tests) + ".ffttest");
            const int k = conv_fast.required_size(size, size);
            FFT<double>::vec_cmplx a(k), b(k);
            for (int j = 0; j < size; ++j) a[j] = dist_values_large(gen);
            for (int j = 0; j < size; ++j) b[j]= dist_values_large(gen);
            os << size << " " << size << "\n";
            dump_vec(os, a, size);
            dump_vec(os, b, size);
            conv_fast.convolve(a, b, 2 * size - 1);
            dump_vec(os, a, 2 * size - 1);
            size <<= 1;
            std::cout << "OK done!\n\n";
        }
    }
    return 0;
}
