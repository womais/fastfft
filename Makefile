CC=g++
CFLAGS=-std=c++17 -I. -fopenmp -march=native -O3

fastfft: main.o avx_cmplx.o
	$(CC) $(CFLAGS) -ofastfft avx_cmplx.o main.o 
avx_cmplx.o: avx_cmplx.cpp
	$(CC) $(CFLAGS) avx_cmplx.cpp -c -o avx_cmplx.o
main.o: main.cpp
	$(CC) $(CFLAGS) main.cpp -c -o main.o
clean:
	rm *.o fastfft
