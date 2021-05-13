#ifndef GPU_TOOLS_H
#define GPU_TOOLS_H

void initialize_gpu_data(int N, void* data);
void initialize_gpu_precomp(size_t N, void *data);
void finish_gpu_data(size_t N);
void run_gpu_pass(int len, int lg_len, int n);
void run_gpu_pass_inv(int len, int lg_len, int n);


#endif
