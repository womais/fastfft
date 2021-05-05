cmake_minimum_required(VERSION 3.19)
project(fastfft)

link_directories(/usr/local/lib)
include_directories("." "/usr/local/include" "/usr/local/opt/llvm/include")
link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib")
set(CMAKE_CXX_COMPILER /usr/local/opt/llvm/bin/clang++)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fopenmp -O3")

add_executable(fastfft main.cpp Convolution.h fft/FFT.h fft/RootsOfUnity.h ConvolutionSlow.h)

