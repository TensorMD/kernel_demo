#include "mkl.h"
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include "math.h"
#include "utils.h"

// #define STRONG_SCALING

extern char *optarg;
extern int optopt;

long long a = 30000, b = 1, c = 200, d = 20, k = 64, m = 4, x = 3;

void kernel4(double *M, double *dP, double *U, double *dR, double *F1,
             double *dH) {
  cblas_dgemm_batch_strided(CblasRowMajor, CblasNoTrans, CblasTrans, c, k, d, 1,
                            M, d, c * d, dP, d, k * d, 0, U, k, k * c, a * b);

  for (int i = 0; i < a * b * c; i++) {
    double temp = cblas_ddot(k, U + i * k, 1, dH + i * k, 1);
    int pos = i * 3;
    F1[pos + 0] = temp * dR[pos];
    F1[pos + 1] = temp * dR[pos + 1];
    F1[pos + 2] = temp * dR[pos + 2];
  }
}

void kernel4_masked(double *M, double *dP, double *U, double *dR, double *F1,
                    double *dH, int *mask)
{
  cblas_dgemm_batch_strided(CblasRowMajor, CblasNoTrans, CblasTrans, c, k, d, 1,
                            M, d, c * d, dP, d, k * d, 0, U, k, k * c, a * b);
  int abc;
  double dM[3 * d];
  double rij, rijinv;
  for (int ia = 0; ia < a; ia++) {
    for (int ib = 0; ib < b; ib++) {
      for (int ic = 0; ic < c; ic++) {
        abc = ia * b * c + ib * c + ic;
        if (mask[abc] == 0) {
          break;
        }
        double temp = cblas_ddot(k, U + abc * k, 1, dH + abc * k, 1);
        int pos = abc * 3;
        F1[pos + 0] = temp * dR[pos + 0];
        F1[pos + 1] = temp * dR[pos + 1];
        F1[pos + 2] = temp * dR[pos + 2];
      }
    }
  }
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

  double *D;
  double *R;
  double *drdrx;
  double *dM;
  int *mask;
  long long R_size = a * b * c;
  long long D_size = a * b * c * x;

  R = (double *) malloc(R_size * sizeof(double));
  D = (double *) malloc(D_size * sizeof(double));
  drdrx = (double *) malloc(D_size * sizeof(double));
  mask = (int *)malloc(R_size * sizeof(int));

  srand(time(0));
  for (long long ia = 0; ia < a; ia++) {
    for (long long ib = 0; ib < b; ib++) {
      long long rnd = rand() % (c / 2) + c / 2;
      for (long long ic = 0; ic < c; ic++) {
        mask[ia * b * c + ib * c + ic] = 0;
        if (ic < rnd) {
          mask[ia * b * c + ib * c + ic] = 1;
        }
      }
    }
  }

  for (long long i = 0; i < R_size; i++) {
    if (mask[i] == 0) {
      D[i * 3 + 0] = 0;
      D[i * 3 + 1] = 0;
      D[i * 3 + 2] = 0;
    } else {
      D[i * 3 + 0] = randfrom(1.0, 3.0);
      D[i * 3 + 1] = randfrom(1.0, 3.0);
      D[i * 3 + 2] = randfrom(1.0, 3.0);
    }
  }

  for (long long i = 0; i < a * b * c; i++) {
    if (mask[i] == 0) {
      R[i] = 0;
      drdrx[i * 3 + 0] = 0;
      drdrx[i * 3 + 1] = 0;
      drdrx[i * 3 + 2] = 0;
      continue;
    }
    double dx = D[i * 3 + 0];
    double dy = D[i * 3 + 1];
    double dz = D[i * 3 + 2];
    double rij = sqrt(dx * dx + dy * dy + dz * dz);
    R[i] = rij;
    drdrx[i * 3 + 0] = dx / rij;
    drdrx[i * 3 + 1] = dy / rij;
    drdrx[i * 3 + 2] = dz / rij;
  }

  double *dH;
  double *M;
  double *dP;
  double *U;
  double *F1;
  double *F1_fused;

  long long M_size = a * b * c * d;
  long long H_size = a * b * c * k;
  long long P_size = a * b * k * d;
  long long U_size = a * b * c * k;
  long long F_size = a * b * c * x;

  dH = (double *)malloc(H_size * sizeof(double));
  M = (double *)malloc(M_size * sizeof(double));
  dP = (double *)malloc(P_size * sizeof(double));
  U = (double *)malloc(U_size * sizeof(double));
  F1 = (double *)malloc(F_size * sizeof(double));
  F1_fused = (double *)malloc(F_size * sizeof(double));

  for (long long i = 0; i < F_size; i++) {
    F1[i] = 0.0;
    F1_fused[i] = 0.0;
  }
  for (long long i = 0; i < H_size; i++) {
    dH[i] = randfrom(0.0, 1.0);
  }
  for (long long i = 0; i < M_size; i++) {
    M[i] = randfrom(0.0, 1.0);
  }
  for (long long i = 0; i < P_size; i++) {
    dP[i] = randfrom(0.0, 1.0);
  }

  kernel4(M, dP, U, drdrx, F1, dH);
  kernel4_masked(M, dP, U, drdrx, F1_fused, dH, mask);

  double diff = 0.0;
  for (long long i = 0; i < F_size; i++) {
    diff += fabs(F1[i] - F1_fused[i]);
  }

  if (world_rank == 0) {
    printf("Diff of kernel 4 and fused kernel 4: %f\n", diff);
  }

  double start, end;
  int iter = 5;

  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();

  for (int i = 0; i < iter; i++) {
    kernel4(M, dP, U, drdrx, F1, dH);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  end = MPI_Wtime();

  if (world_rank == 0) {
    printf("Kernel4 with %d processes computing time %f sec, ", world_size,
           (end - start) / iter);
    printf("using a=%d, b=%d, c=%d, d=%d, k=%d, m=%d, x=%d.\n", a, b, c, d, k,
           m - 1, x);
  }

  free(M);
  free(dH);
  free(dP);
  free(U);
  free(drdrx);
  free(F1);
  free(F1_fused);
  free(D);
  free(mask);
  free(R);

  MPI_Finalize();
  return 0;
}
