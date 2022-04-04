#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define thrd_num 12
#define MAX_THREAD_NUM 100

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
  long partial_sum[MAX_THREAD_NUM] = {0};
  long trunk_size = (n / thrd_num) + 1;

  omp_set_num_threads(thrd_num);
  #pragma omp parallel
  {
    #pragma omp for
    for(long i = 0; i < thrd_num; i++){
      long str_pos = i * trunk_size;
      long end_pos = (i+1) * trunk_size;

      end_pos = end_pos > n ? n : end_pos;
      prefix_sum[str_pos] = 0;
      if (str_pos > 0) prefix_sum[str_pos] = A[str_pos-1];

      for(long j = str_pos+1; j < end_pos; j++){
        prefix_sum[j] = prefix_sum[j-1] + A[j-1];
      }
      partial_sum[i+1] = prefix_sum[end_pos-1];
    }

    #pragma omp single
    {
      for(long i = 1; i < thrd_num; i++) 
        partial_sum[i] = partial_sum[i-1] + partial_sum[i];
    }

    #pragma omp for
    for(long i = 0; i < thrd_num; i++){
      long str_pos = i * trunk_size;
      long end_pos = (i+1) * trunk_size;
      end_pos = end_pos > n ? n : end_pos;

      for(long j = str_pos; j < end_pos; j++){
        prefix_sum[j] = prefix_sum[j] + partial_sum[i];
      }
    }
  }
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
