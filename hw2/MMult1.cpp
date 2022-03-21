// g++ -fopenmp -O3 -march=native MMult1.cpp && ./a.out

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

#define BLOCK_SIZE 256

// Note: matrices are stored in column major order; i.e. the array elements in
// the (m x n) matrix C are stored in the sequence: {C_00, C_10, ..., C_m0,
// C_01, C_11, ..., C_m1, C_02, ..., C_0n, C_1n, ..., C_mn}
void MMult0(long m, long n, long k, double *a, double *b, double *c) {
  for (long j = 0; j < n; j++) {
    for (long p = 0; p < k; p++) {
      for (long i = 0; i < m; i++) {
        double A_ip = a[i+p*m];
        double B_pj = b[p+j*k];
        double C_ij = c[i+j*m];
        C_ij = C_ij + A_ip * B_pj;
        c[i+j*m] = C_ij;
      }
    }
  }
}

void MMult1_OpenMP(long m, long n, long k, double *a, double *b, double *c) {
  #pragma omp for
  for (long j = 0; j < n; j++) {
    for (long p = 0; p < k; p++) {
      for (long i = 0; i < m; i++) {
        double A_ip = a[i+p*m];
        double B_pj = b[p+j*k];
        double C_ij = c[i+j*m];
        C_ij = C_ij + A_ip * B_pj;
        c[i+j*m] = C_ij;
      }
    }
  }
}

void MMult1_blocking(long m, long n, long k, double *a, double *b, double *c) {
  // TODO: See instructions below
  long m_Block = m /  BLOCK_SIZE;
  long n_Block = n /  BLOCK_SIZE;
  long k_Block = k /  BLOCK_SIZE;
  double a_Block[BLOCK_SIZE * BLOCK_SIZE];
  double b_Block[BLOCK_SIZE * BLOCK_SIZE];
  double c_Block[BLOCK_SIZE * BLOCK_SIZE];

  for(long j = 1; j <= n_Block; j++){
    for(long i = 1; i <= m_Block; i++){
      // load blocks of C into fast memory
      for(long col = 0; col < BLOCK_SIZE; col++){
        for(long row = 0; row < BLOCK_SIZE; row++){
          long abs_row = (i-1) * BLOCK_SIZE + row;
          long abs_col = (j-1) * BLOCK_SIZE + col;
          c_Block[row + col * BLOCK_SIZE] = c[abs_row + abs_col * m];
        }
      }

      for(long p = 1; p <= k_Block; p++){
        // load blocks of A, B into fast memory
        for(long col = 0; col < BLOCK_SIZE; col++){
          for(long row = 0; row < BLOCK_SIZE; row++){
            long a_row = (i-1) * BLOCK_SIZE + row;
            long a_col = (p-1) * BLOCK_SIZE + col;
            a_Block[row + col * BLOCK_SIZE] = a[a_row + a_col * m];

            long b_row = (p-1) * BLOCK_SIZE + row;
            long b_col = (j-1) * BLOCK_SIZE + col;
            b_Block[row + col * BLOCK_SIZE] = b[b_row + b_col * m];
          }
        }
        // block matrix multiply computing
        for (long x = 0; x < BLOCK_SIZE; x++) {
          for (long y = 0; y < BLOCK_SIZE; y++) {
            for (long z = 0; z < BLOCK_SIZE; z++) {
              c_Block[z+x*BLOCK_SIZE] = c_Block[z+x*BLOCK_SIZE] + a_Block[z+y*BLOCK_SIZE] * b_Block[y+x*BLOCK_SIZE];
            }
          }
        }
      }
      // store blocks of C back
      for(long col = 0; col < BLOCK_SIZE; col++){
        for(long row = 0; row < BLOCK_SIZE; row++){
          long abs_row = (i-1) * BLOCK_SIZE + row;
          long abs_col = (j-1) * BLOCK_SIZE + col;
          c[abs_row + abs_col * m] = c_Block[row + col * BLOCK_SIZE];
        }
      }
    }
  }

}

int main(int argc, char** argv) {
  const long PFIRST = BLOCK_SIZE;
  const long PLAST = 2000;
  const long PINC = std::max(50/BLOCK_SIZE,1) * BLOCK_SIZE; // multiple of BLOCK_SIZE

  printf(" Dimension       Time    Gflop/s       GB/s        Error    Method\n");
  for (long p = PFIRST; p < PLAST; p += PINC) {
    long m = p, n = p, k = p;
    long NREPEATS = 1e9/(m*n*k)+1;
    double* a = (double*) aligned_malloc(m * k * sizeof(double)); // m x k
    double* b = (double*) aligned_malloc(k * n * sizeof(double)); // k x n
    double* c = (double*) aligned_malloc(m * n * sizeof(double)); // m x n
    double* c_ref = (double*) aligned_malloc(m * n * sizeof(double)); // m x n

    // Initialize matrices
    for (long i = 0; i < m*k; i++) a[i] = drand48();
    for (long i = 0; i < k*n; i++) b[i] = drand48();
    for (long i = 0; i < m*n; i++) c_ref[i] = 0;
    for (long i = 0; i < m*n; i++) c[i] = 0;
    double time, flops, bandwidth, max_err;
    Timer t;

    // Origin
    // t.tic();
    for (long rep = 0; rep < NREPEATS; rep++) { // Compute reference solution
      MMult0(m, n, k, a, b, c_ref);
    }
    // time = t.toc();
    // flops = (2.0 * m * n * k * NREPEATS) / (time * 1.0e9); // TODO: calculate from m, n, k, NREPEATS, time
    // bandwidth = (2.0 * m * n * (k+1) * NREPEATS) / (time * 1.0e9); // TODO: calculate from m, n, k, NREPEATS, time
    // printf("%10d %10f %10f %10f", p, time, flops, bandwidth);

    // max_err = 0;
    // for (long i = 0; i < m*n; i++) max_err = std::max(max_err, fabs(c[i] - c_ref[i]));
    // printf(" %10e    Origin\n", max_err);

    // Blocking Version
    t.tic();
    for (long rep = 0; rep < NREPEATS; rep++) {
      MMult1_blocking(m, n, k, a, b, c);
    }
    time = t.toc();
    flops = (2.0 * m * n * k * NREPEATS) / (time * 1.0e9); // TODO: calculate from m, n, k, NREPEATS, time
    bandwidth = (2.0 * m * n * (k+1) * NREPEATS) / (time * 1.0e9); // TODO: calculate from m, n, k, NREPEATS, time
    printf("%10d %10f %10f %10f", p, time, flops, bandwidth);

    max_err = 0;
    for (long i = 0; i < m*n; i++) max_err = std::max(max_err, fabs(c[i] - c_ref[i]));
    printf(" %10e  Blocking\n", max_err);

    //OpenMP Version
    for (long i = 0; i < m*n; i++) c[i] = 0;
    t.tic();
    for (long rep = 0; rep < NREPEATS; rep++) {
      MMult1_OpenMP(m, n, k, a, b, c);
    }
    time = t.toc();
    flops = (2.0 * m * n * k * NREPEATS) / (time * 1.0e9); // TODO: calculate from m, n, k, NREPEATS, time
    bandwidth = (2.0 * m * n * (k+1) * NREPEATS) / (time * 1.0e9); // TODO: calculate from m, n, k, NREPEATS, time
    printf("%10d %10f %10f %10f", p, time, flops, bandwidth);

    max_err = 0;
    for (long i = 0; i < m*n; i++) max_err = std::max(max_err, fabs(c[i] - c_ref[i]));
    printf(" %10e    OpenMP\n", max_err);

    aligned_free(a);
    aligned_free(b);
    aligned_free(c);
  }

  return 0;
}

// * Using MMult0 as a reference, implement MMult1 and try to rearrange loops to
// maximize performance. Measure performance for different loop arrangements and
// try to reason why you get the best performance for a particular order?
//
//
// * You will notice that the performance degrades for larger matrix sizes that
// do not fit in the cache. To improve the performance for larger matrices,
// implement a one level blocking scheme by using BLOCK_SIZE macro as the block
// size. By partitioning big matrices into smaller blocks that fit in the cache
// and multiplying these blocks together at a time, we can reduce the number of
// accesses to main memory. This resolves the main memory bandwidth bottleneck
// for large matrices and improves performance.
//
// NOTE: You can assume that the matrix dimensions are multiples of BLOCK_SIZE.
//
//
// * Experiment with different values for BLOCK_SIZE (use multiples of 4) and
// measure performance.  What is the optimal value for BLOCK_SIZE?
//
//
// * Now parallelize your matrix-matrix multiplication code using OpenMP.
//
//
// * What percentage of the peak FLOP-rate do you achieve with your code?
//
//
// NOTE: Compile your code using the flag -march=native. This tells the compiler
// to generate the best output using the instruction set supported by your CPU
// architecture. Also, try using either of -O2 or -O3 optimization level flags.
// Be aware that -O2 can sometimes generate better output than using -O3 for
// programmer optimized code.
