CC=g++
CFLAGS=-std=c++17 -I. -fopenmp -march=native -O3

all: fastfft generator 

.PHONY: clean	
fastfft: main.o avx_cmplx.o fft/FFT.h fft/FFTPrecomp.h Complex.h
	$(CC) $(CFLAGS) -ofastfft avx_cmplx.o main.o 
generator: generator.cpp avx_cmplx.cpp fft/FFT.h fft/FFTPrecomp.h Complex.h
	$(CC) $(CFLAGS) generator.cpp avx_cmplx.cpp -ogenerator
avx_cmplx.o: avx_cmplx.cpp
	$(CC) $(CFLAGS) avx_cmplx.cpp -c -o avx_cmplx.o
main.o: main.cpp
	$(CC) $(CFLAGS) main.cpp -c -o main.o
clean:
	rm -rf *.o
