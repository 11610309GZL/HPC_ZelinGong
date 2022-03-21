#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define N          1000
#define Max_N      10000
#define Max_K      100
#define Decay_Thre 1e-6
#define isPrintResidual 0

static inline double norm(double **vec){
    double result = 0.0;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            result += vec[i][j] * vec[i][j];
        }
    }
    return sqrt(result);
}

int main(int argc, char *argv[]){
    clock_t str_t, end_t;
    // Print N size
    printf("Using input N = %d\n", N);

    str_t = clock();
    // Memory Allocation
    double **u = (double**) malloc(N * sizeof(double*));
    for(int i = 0; i < N; i++){
        u[i] = (double*) malloc(N * sizeof(double));
    }

    double **u_next = (double**) malloc(N * sizeof(double*));
    for(int i = 0; i < N; i++){
        u_next[i] = (double*) malloc(N * sizeof(double));
    }

    double **f = (double**) malloc(N * sizeof(double*));
    for(int i = 0; i < N; i++){
        f[i] = (double*) malloc(N * sizeof(double));
    }

    double **res = (double**) malloc(N * sizeof(double*));
    for(int i = 0; i < N; i++){
        res[i] = (double*) malloc(N * sizeof(double));
    }
    double  h = 1.0 / (1 + N);
    int k = 0;
    double ini_res, cur_res;

    #ifdef _OPENMP
    printf("Using OpenMP\n");
    // omp_set_num_threads(4);
    #pragma omp parallel 
    #endif
{
    // Initialization
    double tmp_sum = 0.0;
    #pragma omp for collapse(2)
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            u[i][j] = 0.0;
        }
    }

    #pragma omp for collapse(2)
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            u_next[i][j] = 0.0;
        }
    }

    #pragma omp for collapse(2)
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            f[i][j] = 1.0;
        }
    }

    #pragma omp for collapse(2)
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            res[i][j] = 0.0;
        }
    }

    #pragma omp for collapse(2)
    for(int i = 1; i < N-1; i++){
        for(int j = 1; j < N-1; j++){
            tmp_sum = (4 * u[i][j]) - u[i][j+1] - u[i][j-1] - u[i+1][j] - u[i-1][j];
            tmp_sum *= (1 / h) * (1 / h);
            res[i][j] = tmp_sum - f[i][j];
        }
    }
}
    ini_res = norm(res);
    printf("ini residual: %f\n",ini_res);
    cur_res = ini_res;

    // Jacobi-2D Algorithm
    while((cur_res / ini_res) > Decay_Thre && k < Max_K){
        // Compute by formula
        #ifdef _OPENMP
        #pragma omp parallel
        #endif
        {
        double tmp_sum = 0.0;
        #pragma omp for collapse(2)
        for(int i = 1; i < N-1; i++){
            for(int j = 1; j < N-1; j++){
                tmp_sum =  (1 / 4.0) * ((h * h * f[i][j]) - u[i-1][j] - u[i][j-1] - u[i+1][j] - u[i][j+1]);
                u_next[i][j] = tmp_sum;
            }
        }
        // Updating to u[][]
        #pragma omp for collapse(2)
        for(int i = 1; i < N-1; i++){
            for(int j = 1; j < N-1; j++){
                u[i][j] = u_next[i][j];
            }
        }
        // Compute residual
        #pragma omp for collapse(2)
        for(int i = 1; i < N-1; i++){
            for(int j = 1; j < N-1; j++){
                tmp_sum = (4 * u[i][j]) - u[i][j+1] - u[i][j-1] - u[i+1][j] - u[i-1][j];
                tmp_sum *= (1 / h) * (1 / h);
                res[i][j] = tmp_sum - f[i][j];
            }
        }
        }
        cur_res = norm(res);
        if(isPrintResidual) printf("Iteration %d, cur residual: %.6f\n", k, cur_res);
        k++;
    }

    end_t = clock();
    double tol_t = (double)(end_t - str_t) / CLOCKS_PER_SEC;
    printf("Total Time: %f\n", tol_t);
    printf("---------------END------------\n");

    for(int i = 0; i < N; i++){
        free(u[i]);
    }
    free(u);

    for(int i = 0; i < N; i++){
        free(u_next[i]);
    }
    free(u_next);

    for(int i = 0; i < N; i++){
        free(f[i]);
    }
    free(f);

    for(int i = 0; i < N; i++){
        free(res[i]);
    }
    free(res);

    return 0;
}