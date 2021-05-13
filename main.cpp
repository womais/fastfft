#include <algorithm>
#include <iostream>
#include <fstream>
#include <thread>
#include <set>
#include <fft/FFT.h>
#include <filesystem>

namespace fs = std::filesystem;

using std::vector;
using fftvec = FFT<double>::vec_cmplx; // aligned vector type.

void do_test_set(const std::vector<std::pair<std::string, std::string> >& file_paths,
                 const FFT<double>& conv) {
    std::cout << "Starting tests using following implementation: \n";
    std::cout << conv.name() << "\n";
    std::cout << "************************\n";
    bool fail = false;

    for (const auto& [name, p] : file_paths) {
        auto dot = p.find_last_of(".");
        if (dot != std::string::npos &&
            p.substr(dot + 1) == "ffttest") {
            std::cout << "Doing test " << name << "...\n";
            std::ifstream stream(p);
            int n, m;
            stream >> n >> m;
            int need = n + m - 1;
            int k = 1;
            while (k < n + m)
                k <<= 1;
            fftvec a(k), b(k);
            vector<long long> expected(need);
            for (int i = 0; i < n; ++i) { int z; stream >> z; a[i] = z; }
            for (int i = 0; i < m; ++i) { int z; stream >> z; b[i] = z; }
            for (int i = 0; i < need; ++i) stream >> expected[i];
            double elapsed = conv.timed_convolve(a, b, n + m - 1);
            bool all_correct = true;
            for (int i = 0; i < n + m - 1; ++i)
                all_correct &= (std::llround(a[i].real()) == expected[i]);
            if (all_correct) {
                std::cout << "OK, correct result! ";
            } else {
                fail = true;
                std::cout << "Incorrect answer! ";
            }
            std::cout << "Time taken was " << elapsed << " ms.\n\n";
        }
    }
    std::cout << "Finished all tests.\n";
    if (!fail)
        std::cout << "All tests passed!\n";
    std::cout << "************************\n";
}
int main(int argc, const char* argv[]) {
    if (argc < 3 || argc > 6) {
        std::cout << "Incorrect number of arguments.\n";
        exit(1);
    }
    vector<std::pair<std::string, std::string> > file_paths_sample;
    vector<std::pair<std::string, std::string> > file_paths_rand;
    vector<std::pair<std::string, std::string> > file_paths_randlarge;
    for (const auto& entry : fs::directory_iterator(argv[1])) {
        std::string filename = entry.path().filename().c_str();
        if (filename.find("random-large") != std::string::npos) {
            file_paths_randlarge.emplace_back(entry.path().filename().c_str(), entry.path().c_str());
        } else if (filename.find("random") != std::string::npos) {
            file_paths_rand.emplace_back(entry.path().filename().c_str(), entry.path().c_str());
        } else {
            file_paths_sample.emplace_back(entry.path().filename().c_str(), entry.path().c_str());
        }
    }
    auto order = [&](std::vector<std::pair<std::string, std::string>>& path_list) {
        std::sort(path_list.begin(), path_list.end(), [&](const auto& p1, const auto& p2) {
            if (p1.first.size() != p2.first.size()) return p1.first.size() < p2.first.size();
            return p1.first < p2.first;
        });
    };
    std::cout << "Initializing FFT...\n";
    FFTPrecomp<double>::initialize_to(1 << 27, 2);
    std::cout << "FFT Initialized!\n\n";
    order(file_paths_sample);
    order(file_paths_rand);
    order(file_paths_randlarge);
    std::set<std::string> flags;
    int cores = 1;
    for (int i = 2; i < argc; ++i) {
        flags.insert(argv[i]);
        std::string arg(argv[i]);
        if (arg.size() > 8 && arg.substr(0, 8) == "--procs=") {
            cores = std::stoi(arg.substr(8));
        }
    }
    if (flags.count("--sample")) {
        std::cout << "Starting samples with " << cores << " cores.\n\n";
        do_test_set(file_paths_sample, FFTIterative(cores));
        std::cout << "\n\n";
    }
    if (flags.count("--rand")) {
        std::cout << "Starting random cases with " << cores << " cores.\n\n";
        do_test_set(file_paths_rand, FFTIterative(cores));
        std::cout << "\n\n";
    }
    if (flags.count("--randlarge")) {
        std::cout << "Starting large random cases with " << cores << " cores.\n\n";
        do_test_set(file_paths_randlarge, FFTIterative(cores));
        std::cout << "\n\n";
    }
    return 0;
}
