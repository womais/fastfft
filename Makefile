CC=g++-10

CFLAGS=-std=c++17 -I. -lcuda -lcudart -fopenmp -march=native -O3
HEADERS=fft/FFT.h fft/FFTPrecomp.h Complex.h

all: fastfft

.PHONY: clean

cuda/fft_pass.o: cuda/fft_pass.cu
	nvcc -O3 -c -o cuda/fft_pass.o cuda/fft_pass.cu
fastfft: main.o avx_cmplx.o cuda/fft_pass.o
	nvcc -O3 -Xcompiler -fopenmp -o fastfft avx_cmplx.o main.o cuda/fft_pass.o
generator: generator.o avx_cmplx.o cuda/fft_pass.o
	nvcc -O3 -Xcompiler -fopenmp -o generator generator.o avx_cmplx.o cuda/fft_pass.o
generator.o: generator.cpp $(HEADERS)
	$(CC) $(CFLAGS) generator.cpp -c -o generator.o
avx_cmplx.o: avx_cmplx.cpp $(HEADERS)
	$(CC) $(CFLAGS) avx_cmplx.cpp -c -o avx_cmplx.o
main.o: main.cpp $(HEADERS)
	$(CC) $(CFLAGS) main.cpp -c -o main.o
clean:
	rm -rf *.o
