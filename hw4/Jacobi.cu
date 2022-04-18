#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string>

#define BLOCK_SIZE 1024
#define N          1000
#define Max_N      10000
#define Max_K      100
#define Decay_Thre 1e-6
#define isPrintResidual 0

__global__
void Jacobin_2D_GPU(double *u, double *u_next, double *f, double *res){

  int idx =  blockIdx.x * blockDim.x + threadIdx.x;
  int i =  idx / N;
  int j =  idx - (i * N);
  double h = 1.0 / (1 + N);
  double tmp_sum = 0.0;

  if( (i > 0) && (i < N) && (j > 0) && (j < N))
    u_next[idx] = (1 / 4.0) * ((h * h * f[idx]) - u[idx-N] - u[idx-1] - u[idx+N] - u[idx+1]);
    __syncthreads();
    u[idx] = u_next[idx];
    __syncthreads();
    tmp_sum = (4 * u[idx]) - u[idx+1] - u[idx-1] - u[idx+N] - u[idx-N];
    tmp_sum *= (1 / h) * (1 / h);
    res[idx] = tmp_sum - f[idx];
}

void Jacobin_2D(double *u, double *u_next, double *f, double *res){
    double h = 1.0 / (1 + N);
    #pragma omp parallel
    {
        double tmp_sum = 0.0;
        #pragma omp for collapse(2)
        for(int i = 1; i < N-1; i++){
            for(int j = 1; j < N-1; j++){
                tmp_sum =  (1 / 4.0) * ((h * h * f[i*N +j]) - u[(i-1)*N +j] - u[i*N +j-1] - u[(i+1)*N +j] - u[i*N +j+1]);
                u_next[i*N +j] = tmp_sum;
            }
        }
        // Updating to u[][]
        #pragma omp for collapse(2)
        for(int i = 1; i < N-1; i++){
            for(int j = 1; j < N-1; j++){
                u[i*N +j] = u_next[i*N +j];
            }
        }
        // Compute residual
        #pragma omp for collapse(2)
        for(int i = 1; i < N-1; i++){
            for(int j = 1; j < N-1; j++){
                tmp_sum = (4 * u[i*N +j]) - u[i*N +j+1] - u[i*N +j-1] - u[(i+1)*N +j] - u[(i-1)*N +j];
                tmp_sum *= (1 / h) * (1 / h);
                res[i*N +j] = tmp_sum - f[i*N +j];
            }
        }
    }
}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

static inline double norm(double *vec){
    double result = 0.0;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            result += vec[i*N +j] * vec[i*N+j];
        }
    }
    return sqrt(result);
}

int main(){
    double *u, *u_next, *u_gpu, *f, *res;
    cudaMallocHost((void**)&u,      N * N * sizeof(double));
    cudaMallocHost((void**)&u_next, N * N * sizeof(double));
    cudaMallocHost((void**)&u_gpu,  N * N * sizeof(double));
    cudaMallocHost((void**)&f,      N * N * sizeof(double));
    cudaMallocHost((void**)&res,    N * N * sizeof(double));
    #pragma omp parallel for
    for(long i = 0; i < N * N; i++){
        u[i] = 0;
        u_next[i] = 0;
        u_gpu[i] = 0;
        f[i] = 1;
        res[i] = 0;
    }

    double h = 1.0 / (1 + N);
    double ini_res, cur_res;
    double tmp_sum = 0;
    for(int i = 1; i < N-1; i++){
        for(int j = 1; j < N-1; j++){
            tmp_sum = (4 * u[i*N+j]) - u[i*N+j+1] - u[i*N+j-1] - u[(i+1)*N+j] - u[(i-1)*N+j];
            tmp_sum *= (1 / h) * (1 / h);
            res[i*N+j] = tmp_sum - f[i*N+j];
        }
    }
    ini_res = norm(res);
    cur_res = ini_res;

    int k = 0;
    double tt = omp_get_wtime();
    while((cur_res / ini_res) > Decay_Thre && k < Max_K){
        Jacobin_2D(u, u_next, f, res);
        cur_res = norm(res);
        if(isPrintResidual) printf("Iteration %d, cur residual: %.6f\n", k, cur_res);
        k++;
    }
    double tol_t = omp_get_wtime() - tt;
    printf("Total Time: %.6f\n", tol_t);

    k = 0;
    double *u_d, *u_next_d, *f_d, *res_d;
    cudaMalloc(&u_d,      N*N*sizeof(double));
    cudaMalloc(&u_next_d, N*N*sizeof(double));
    cudaMalloc(&f_d,      N*N*sizeof(double));
    cudaMalloc(&res_d,    N*N*sizeof(double));

    tt = omp_get_wtime();
    // At this time u_gpu is zeros matirx.
    cudaMemcpyAsync(u_d,      u_gpu,  N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(u_next_d, u_gpu,  N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(f_d,      f,      N*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(res_d,    u_gpu,  N*N*sizeof(double), cudaMemcpyHostToDevice);
    while((cur_res / ini_res) > Decay_Thre && k < Max_K){
        Jacobin_2D_GPU<<<N*N/1024, 1024>>>(u_d, u_next_d, f_d, res_d);
        cudaMemcpyAsync(res,  res_d,  N*N*sizeof(double), cudaMemcpyDeviceToHost);
        cur_res = norm(res);
        if(isPrintResidual) printf("Iteration %d, cur residual: %.6f\n", k, cur_res);
        k++;
    }
    cudaMemcpyAsync(u_d,  u_gpu,  N*N*sizeof(double), cudaMemcpyDeviceToHost);
    tol_t = omp_get_wtime() - tt;
    printf("Total Time GPU: %.6f\n", tol_t);

    double err = 0.0;
    for (long i = 0; i < N*N; i++) err += fabs(u[i] - u_gpu[i]);
    printf("Error = %f\n", err);

    cudaFree(u_d);
    cudaFree(u_next_d);
    cudaFree(f_d);
    cudaFree(res_d);

    cudaFreeHost(u);
    cudaFreeHost(u_next);
    cudaFreeHost(u_gpu);
    cudaFreeHost(f);
    cudaFreeHost(res);

    return 0;
}