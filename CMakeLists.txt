cmake_minimum_required(VERSION 3.19)
project(fastfft)

link_directories(/usr/local/lib)
include_directories("." "/usr/local/include" "/usr/local/opt/llvm/include")
link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib")
set(CMAKE_CXX_COMPILER /usr/local/opt/llvm/bin/clang++)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fopenmp -march=native -O3")

add_executable(fastfft main.cpp avx_cmplx.cpp)
add_executable(generator generator.cpp avx_cmplx.cpp)
add_executable(stringdemo string-matching/demo.cpp avx_cmplx.cpp)