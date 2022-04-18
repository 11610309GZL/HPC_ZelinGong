#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

#define BLOCK_SIZE 1024

void vec_prod(double* c, const double* a, const double* b, long N){
  double sum = 0;
  #pragma omp parallel for reduction(+:sum)
  for(long i = 0; i < N; i++){
      sum += a[i] * b[i];
  }
  c[0] = sum;
}

__global__
void vec_prod_gpu(double* c, const double* a, const double* b, long N, int offset){
//   long blockNum = (N / BLOCK_SIZE) + 1;
  __shared__ double smem[BLOCK_SIZE];
  smem[threadIdx.x] = 0;

  int idx =  blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) smem[threadIdx.x] = a[offset +idx] * b[idx];
  __syncthreads();

  for(unsigned int s = blockDim.x/2; s > 0; s >>= 1){
      if(threadIdx.x < s)
        smem[threadIdx.x] += smem[threadIdx.x + s];
      __syncthreads();
  }

  if(threadIdx.x == 0) c[blockIdx.x] = smem[threadIdx.x];
  __syncthreads();

//   for(int s = 1; s < blockNum; s *= 2){
//       if((threadIdx.x == 0) && (threadIdx.x % (2*s) == 0))
//         smem[threadIdx.x] += smem[threadIdx.x + s];
//       __syncthreads();
//   }

}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

int main() {
  //long N = (1UL<<25);
  //const int blockSize = 1024;

  long N = 100000000;
  long blockNum = (N / BLOCK_SIZE) + 1;
  printf("N: %ld\n", N);

  double *x, *y, *z;
  cudaMallocHost((void**)&x, N * sizeof(double));
  cudaMallocHost((void**)&y, N * sizeof(double));
  cudaMallocHost((void**)&z, blockNum * sizeof(double));
  double* z_ref = (double*) malloc(blockNum * sizeof(double));

  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    x[i] = i+2;
    y[i] = 1.0/(i+2);
  }
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < blockNum; i++) {
    z[i] = 0;
    z_ref[i] = 0;
  }

  double tt = omp_get_wtime();
  vec_prod(z_ref, x, y, N);
  printf("CPU Bandwidth = %f GB/s\n", 2*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  double *x_d, *y_d, *z_d;
  cudaMalloc(&x_d, N*sizeof(double));
  cudaMalloc(&y_d, N*sizeof(double));
  cudaMalloc(&z_d, blockNum*sizeof(double));

  tt = omp_get_wtime();
  cudaMemcpyAsync(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(y_d, y, N*sizeof(double), cudaMemcpyHostToDevice);
  vec_prod_gpu<<<N/1024,1024>>>(z_d, x_d, y_d, N, 0);
  cudaMemcpyAsync(z, z_d, blockNum*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("GPU Bandwidth = %f GB/s\n", 2*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  double err = z_ref[0], gpu_sum = 0;
  for (long i = 0; i < blockNum; i++){err -= z[i]; gpu_sum += z[i];} 
  err = fabs(err);
  printf("Error = %f\n", err);
  printf("CPU Result: %f\n", z_ref[0]);
  printf("GPU Result: %f\n", gpu_sum);

  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(z_d);

  cudaFreeHost(x);
  cudaFreeHost(y);
  cudaFreeHost(z);
  free(z_ref);

  long N_mat = 10000;
  blockNum = (N_mat / BLOCK_SIZE) + 1;
  printf("Mat: %ld * %ld\n", N_mat, N_mat);
  double *tmpsum;
  cudaMallocHost((void**)&x, N_mat * N_mat * sizeof(double));
  cudaMallocHost((void**)&y, N_mat * sizeof(double));
  cudaMallocHost((void**)&z, N_mat * sizeof(double));
  cudaMallocHost((void**)&tmpsum, blockNum * sizeof(double));
  z_ref = (double*) malloc(N_mat * sizeof(double));

  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N_mat; i++) {
    for (long j = 0; j < N_mat; j++) {
        x[i*N_mat + j] = 1.0/(i+2);
    }
  }
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N_mat; i++) {
    y[i] = 1.0/(i+2);
    z[i] = 0;
    z_ref[i] = 0;
  }

  tt = omp_get_wtime();
  for(long i = 0; i < N_mat; i++){
    vec_prod(&(z_ref[i]), &(x[i*N_mat]), y, N_mat);
  }
  printf("CPU Bandwidth = %f GB/s\n", N_mat*N_mat*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  double *tmpsum_d;
  cudaMalloc(&x_d, N_mat * N_mat*sizeof(double));
  cudaMalloc(&y_d, N_mat * sizeof(double));
//   cudaMalloc(&z_d, N_mat * sizeof(double));
  cudaMalloc((void**)&tmpsum_d, blockNum * sizeof(double));

  tt = omp_get_wtime();
  cudaMemcpyAsync(x_d, x, N_mat * N_mat*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(y_d, y, N_mat *sizeof(double), cudaMemcpyHostToDevice);
  for(long i = 0; i < N_mat; i++){
    vec_prod_gpu<<<N_mat/1024,1024>>>(tmpsum_d, x_d, y_d, N_mat, i * N_mat);
    cudaMemcpyAsync(tmpsum, tmpsum_d, blockNum*sizeof(double), cudaMemcpyDeviceToHost);
    double sum = 0;
    for(long j = 0; j < blockNum; j++) sum += tmpsum[j];
    z[i] = sum;
    // if(i % 100 == 0) printf("itea %d\n",i);
  }
  cudaDeviceSynchronize();
  printf("GPU Bandwidth = %f GB/s\n", N_mat*N_mat*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  err = z_ref[0];
  for (long i = 0; i < N_mat; i++) err += fabs(z_ref[i] - z[i]);
  printf("Error = %f\n", err);
//   printf("CPU Result: %f\n", z_ref[0]);
//   printf("GPU Result: %f\n", gpu_sum);

  cudaFree(x_d);
  cudaFree(y_d);
  // cudaFree(z_d);
  cudaFree(tmpsum_d);

  cudaFreeHost(x);
  cudaFreeHost(y);
  cudaFreeHost(z);
  cudaFreeHost(tmpsum);
  free(z_ref);

  return 0;
}