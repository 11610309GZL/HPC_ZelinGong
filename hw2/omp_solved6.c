/******************************************************************************
* FILE: omp_bug6.c
* DESCRIPTION:
*   This program compiles and runs fine, but produces the wrong result.
*   Compare to omp_orphan.c.
* AUTHOR: Blaise Barney  6/05
* LAST REVISED: 06/30/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 100

/**
 *  Multiple defined sum may cause false name shadowing,
 *  change it to a global variable 
 **/
float sum;

float a[VECLEN], b[VECLEN];

void dotprod ()
{
int i,tid;
// float sum;

/**
 * We add parallel here, and move the tid get inside the for-loop.
 **/
#pragma omp parallel for reduction(+:sum)
  for (i=0; i < VECLEN; i++)
    {
    tid = omp_get_thread_num();
    sum = sum + (a[i]*b[i]);
    printf("  tid= %d i=%d\n",tid,i);
    }
}


int main (int argc, char *argv[]) {
int i;
// float sum;

for (i=0; i < VECLEN; i++)
  a[i] = b[i] = 1.0 * i;
sum = 0.0;

/**
 * Here the #pragma omp parallel would let different thread excute the
 * function for several times, which is not what we want.
 * We move parallel inside the function instead.
 **/
// #pragma omp parallel shared(sum)
  dotprod();

printf("Sum = %f\n",sum);

}
