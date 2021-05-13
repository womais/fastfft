#include <stdio.h>

struct Complex {
    double real;
    double imag;
};
Complex* a_device = NULL;
Complex* host_mem = NULL;
Complex* device_precomp = NULL;

__global__
void donkey_inv(Complex* precomp,
                Complex* a, 
                int blocks_per_half, 
                int lg_len,
                int num_blocks) {
    const int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk > num_blocks) return;
    const int len = 1 << lg_len;

    // more...
    const int which_half = blk / blocks_per_half;
    const int block_ind = blk % blocks_per_half;
    const int half_start = (which_half << lg_len);
    const int start = half_start + block_ind * 2;

    Complex w[2], u[2], v[2];
    for (int j = 0; j < 2; ++j) {
        w[j] = precomp[len + start - half_start + j];
        u[j] = a[start + j];
        v[j] = a[start + j + (len >> 1)];
        w[j].imag *= -1;
        v[j] = {v[j].real * w[j].real - v[j].imag * w[j].imag,
                v[j].real * w[j].imag + v[j].imag * w[j].real};
        a[start + j].real = u[j].real + v[j].real;
        a[start + j].imag = u[j].imag + v[j].imag;
        a[start + j + (len >> 1)].real = u[j].real - v[j].real;
        a[start + j + (len >> 1)].imag = u[j].imag - v[j].imag;
    }
}

__global__
void donkey(Complex* precomp,
                Complex* a, 
                int blocks_per_half, 
                int lg_len,
                int num_blocks) {
    const int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk > num_blocks) return;
    const int len = 1 << lg_len;

    // more...
    const int which_half = blk / blocks_per_half;
    const int block_ind = blk % blocks_per_half;
    const int half_start = (which_half << lg_len);
    const int start = half_start + block_ind * 2;

    Complex w[2], u[2], v[2];
    for (int j = 0; j < 2; ++j) {
        w[j] = precomp[len + start - half_start + j];
        u[j] = a[start + j];
        v[j] = a[start + j + (len >> 1)];
        v[j] = {v[j].real * w[j].real - v[j].imag * w[j].imag,
                v[j].real * w[j].imag + v[j].imag * w[j].real};
        a[start + j].real = u[j].real + v[j].real;
        a[start + j].imag = u[j].imag + v[j].imag;
        a[start + j + (len >> 1)].real = u[j].real - v[j].real;
        a[start + j + (len >> 1)].imag = u[j].imag - v[j].imag;
    }
}

// costly, but should only do it once...
// this copies over precomputed roots of unity
// to global memory.
extern "C"
{
void initialize_gpu_precomp(size_t N, void* data) {
    cudaMalloc(&device_precomp, N * sizeof(Complex));
    cudaMemcpy(device_precomp, data, N * sizeof(Complex), cudaMemcpyHostToDevice);
}

void initialize_gpu_data(size_t N, void* values) {
    if (a_device == NULL)
        cudaMalloc(&a_device, N * sizeof(Complex));
    host_mem = (Complex*)values;
    cudaMemcpy(a_device, host_mem, N * sizeof(Complex), cudaMemcpyHostToDevice);
}

void finish_gpu_data(size_t N) {
    cudaMemcpy(host_mem, a_device, N * sizeof(Complex), cudaMemcpyDeviceToHost);
    cudaFree(a_device);
    a_device = NULL;
}
void run_gpu_pass(int len, int lg_len, int n) {
    const int num_half_intervals = n >> lg_len;
    const int blocks_per_half = (len >> 2);
    const int num_blocks = blocks_per_half * num_half_intervals; 
    donkey<<<(num_blocks + 255) / 512, 512>>>(device_precomp, a_device, blocks_per_half, lg_len, num_blocks);
    cudaDeviceSynchronize();

}
void run_gpu_pass_inv(int len, int lg_len, int n) {
    const int num_half_intervals = n >> lg_len;
    const int blocks_per_half = (len >> 2);
    const int num_blocks = blocks_per_half * num_half_intervals; 
    donkey_inv<<<(num_blocks + 255) / 512, 512>>>(device_precomp, a_device, blocks_per_half, lg_len, num_blocks);
    cudaDeviceSynchronize();

}

}
