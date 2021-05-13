#include <iostream>
#include <string>
#include <random>
#include <vector>
#include <chrono>
#include <cmath>
#include <omp.h>
#include <fft/FFT.h>

using std::string;
using std::vector;

vector<int> naive_matcher(const string& pattern,
                          const string& target,
                          double* elapsed) {
    auto start = std::chrono::high_resolution_clock::now();
    const int m = target.size(), n = pattern.size();
    vector<int> result;
    for (int i = 0; i <= n - m; ++i) {
        bool ok = true;
        for (int j = 0; ok && j < m; ++j)
            ok &= (target[j] == '*' || target[j] == pattern[i + j]);
        if (ok) {
            result.push_back(i);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    *elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    return result;
}

/* One-time initialization cost. */
static std::array<complex<double>, 26> roots;
static bool initialized = false;

void precompute() {
    const double PI = std::acos(-1.0);
    for (int i = 0; i < 26; ++i) {
        roots[i] = {std::cos(2.0 * PI * i / 26), std::sin(2.0 * PI * i / 26)};
    }
}

vector<int> fft_matcher(const string& pattern,
                        const string& target,
                        int cores,
                        double* elapsed) {

    /* the idea behind this algorithm is as follows:
     * Define polynomial P(x) = sum of alpha_i * x^i, where alpha_j = exp(2 * pi * i * P[j] / 26),
     * where 26 is an upper bound on alphabet size (but this can be changed if needed,
     * nothing special about 26).
     *
     * Also do the same for another poly. T(x), where coefficients are picked from the target.
     * Make sure to reverse T(x), so that convolution gives terms corresponding to "aligning"
     * the target at various points in the pattern.
     *
     * A convolution of m + 0 i denotes a match. Anything else is a mismatch.
     */
    FFTIterative convolver(cores);
    const int n = pattern.size(), m = target.size();
    int wildcards = std::count(target.begin(), target.end(), '*');
    const int need = convolver.required_size(n, m);
    FFT<double>::vec_cmplx va(need);
    FFT<double>::vec_cmplx vb(need);
    for (int i = 0; i < n; ++i)
        va[i] = roots[pattern[i] - 'a'];
    for (int i = 0; i < m; ++i) {
        if (target[i] != '*') {
            vb[m - i - 1] = roots[(26 - (target[i] - 'a')) % 26];
        }
    }
    *elapsed = convolver.timed_convolve(va, vb, n + m - 1);
    vector<int> result;
    for (int i = 0; i <= n - m; ++i) {
        if (fabs(va[i + m - 1].imag() < 1e-4) &&
            fabs(va[i + m - 1].real() - (m - wildcards)) < 1e-4)
        {
            result.push_back(i);
        }
    }
    return result;
}

int main(int argc, const char* argv[]) {
    if (argc < 5 || argc > 6) {
        std::cout << "Invalid input.\n";
        exit(1);
    }
    string pattern(atoi(argv[1]), '\0');
    string target(atoi(argv[2]), '\0');
    int alphabet = atoi(argv[3]);
    double prob_wild = std::stod(argv[4]);
    if (alphabet > 26 || alphabet < 1) {
        std::cout << "Error: " << "alphabet should be between 1 and 26.\n";
        exit(1);
    }
    int cores = 1;
    if (argc == 6)
        cores = atoi(argv[5]);
    std::cout << "Initializing FFT...\n";
    FFTPrecomp<double>::initialize_to(FFTIterative().
            required_size(pattern.size(), target.size()));
    precompute();
    std::cout << "FFT initialized!\n\n";
    std::random_device device;
    std::mt19937 gen(device());
    std::uniform_int_distribution<int> dist_size(0, alphabet - 1);
    std::uniform_real_distribution<double> dist_wildcard(0.0, 1.0);
    std::cout << "Generating strings...\n";
    for (int i = 0; i < pattern.size(); ++i)
        pattern[i] = static_cast<char>('a' + dist_size(gen));
    for (int i = 0; i < target.size(); ++i) {
        if (dist_wildcard(gen) < prob_wild) {
            target[i] = '*';
        } else {
            target[i] = static_cast<char>('a' + dist_size(gen));
        }
    }
    std::cout << "OK, string generation successful! Counting matches...\n";
    double elapsed_kmp, elapsed_fft;
    auto matches_kmp = naive_matcher(pattern, target, &elapsed_kmp);
    auto matches_fft = fft_matcher(pattern, target, cores, &elapsed_fft);
    std::cout << "Naive matcher got " << matches_kmp.size() << " matches in " << elapsed_kmp << " ms.\n";
    std::cout << "FFT got " << matches_fft.size() << " matches in " << elapsed_fft << " ms.\n\n";
    if (matches_kmp == matches_fft) {
        std::cout << "Outputs agreed!\n";
    } else {
        std::cout << "ERROR! Output mismatch.\n";
    }
}
