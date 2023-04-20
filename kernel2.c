#include "mkl.h"
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include "utils.h"

// #define STRONG_SCALING
#define TEST

extern char *optarg;
extern int optopt;

long long a = 30000, b = 1, c = 200, d = 20, k = 64, m = 4, x = 3;

void kernel2(double *dQ, double *T, double *dS, double *P, double *dP) {
#ifdef TEST
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, a * b * k, d, m, 2, dQ,
              m, T, m, 0, dS, d);
#else
  cblas_dgemm_batch_strided(CblasRowMajor, CblasNoTrans, CblasTrans, k, d, m, 2,
                            dQ, m, m * k, T, m, 0, 0, dS, d, d * k, a * b);
#endif
  vdMul(a * b * k * d, dS, P, dP);
}

int main(int argc, char *argv[]) {
  int world_size, world_rank;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int arg_temp;

  while (EOF != (arg_temp = getopt(argc, argv, "a::b::c::k::m::"))) {
    switch (arg_temp) {
    case 'a':
      if (optarg == NULL)
        printf("Legal input should be like -a30000, using default "
               "configuration.\n");
      else
        a = atoi(optarg);
      break;
    case 'b':
      if (optarg == NULL)
        printf(
            "Legal input should be like -b1, using default configuration.\n");
      else
        b = atoi(optarg);
      break;
    case 'c':
      if (optarg == NULL)
        printf(
            "Legal input should be like -c200, using default configuration.\n");
      else
        c = atoi(optarg);
      break;
    case 'k':
      if (optarg == NULL)
        printf(
            "Legal input should be like -k64, using default configuration.\n");
      else
        k = atoi(optarg);
      break;
    case 'm':
      if (optarg == NULL) {
        printf(
            "Legal input should be like -m3, using default configuration.\n");
      } else {
        m = atoi(optarg) + 1;
        if (m < 1 || m > 4) {
          printf("Illegal value of m, using default configureation.\n");
          m = 4;
          d = 20;
        } else if (m == 1)
          d = 1;
        else if (m == 2)
          d = 4;
        else if (m == 3)
          d = 10;
      }
      break;
    case '?':
      printf("Unsupported argument %c, using defualt configuration.\n", optopt);
      break;
    }
  }

#ifdef STRONG_SCALING
  a = a / world_size;
#endif

  double *P;
  double *dP;
  double *dS;
  double *T;
  double *dQ;

  long long P_size = a * b * k * d;
  long long S_size = a * b * k * d;
  long long T_size = d * m;
  long long Q_size = a * b * k * m;

  P = (double *)malloc(P_size * sizeof(double));
  dP = (double *)malloc(P_size * sizeof(double));
  dS = (double *)malloc(S_size * sizeof(double));
  T = (double *)malloc(T_size * sizeof(double));
  dQ = (double *)malloc(Q_size * sizeof(double));

  srand(time(0));
  for (long long i = 0; i < P_size; i++) {
    P[i] = randfrom(0.0, 1.0);
  }
  for (long long i = 0; i < Q_size; i++) {
    dQ[i] = randfrom(0.0, 1.0);
  }
  for (long long i = 0; i < T_size; i++) {
    T[i] = 0;
  }
  T[0] = 1;
  if (m > 1) {
    T[5] = 1;
    T[9] = 1;
    T[13] = 1;
  }
  if (m > 2) {
    T[2] = a;
    T[18] = 1;
    T[22] = 2;
    T[26] = 2;
    T[30] = 1;
    T[34] = 2;
    T[38] = 1;
  }
  if (m > 3) {
    T[7] = b;
    T[11] = b;
    T[15] = b;
    T[43] = 1;
    T[47] = 3;
    T[51] = 3;
    T[55] = 3;
    T[59] = 6;
    T[63] = 3;
    T[67] = 1;
    T[71] = 3;
    T[75] = 3;
    T[79] = 1;
  }

  double start, end;
  int iter = 5;

  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();

  for (int i = 0; i < iter; i++) {
    kernel2(dQ, T, dS, P, dP);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  end = MPI_Wtime();

  if (world_rank == 0) {
    printf("Kernel2 with %d processes computing time %f sec, ", world_size,
           (end - start) / iter);
    printf("using a=%d, b=%d, c=%d, d=%d, k=%d, m=%d, x=%d.\n", a, b, c, d, k,
           m - 1, x);
  }

  free(P);
  free(dP);
  free(dS);
  free(T);
  free(dQ);

  MPI_Finalize();
  return 0;
}
