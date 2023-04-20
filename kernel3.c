#include "mkl.h"
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include "math.h"
#include "utils.h"
#include "calculate_dM.h"

// #define STRONG_SCALING

extern char *optarg;
extern int optopt;

long long a = 1000, b = 1, c = 200, d = 20, k = 64, m = 4, x = 3;

void kernel3(double *H, double *dP, double *V, double *dM, double *F2) {
  cblas_dgemm_batch_strided(CblasRowMajor, CblasNoTrans, CblasNoTrans, c, d, k,
                            1, H, k, c * k, dP, d, k * d, 0, V, d, d * c,
                            a * b);
  cblas_dgemv_batch_strided(CblasRowMajor, CblasNoTrans, x, d, 1, dM, d, d * x,
                            V, 1, d, 0, F2, 1, x, a * b * c);
}

double simd_dot(double *y, double *kernel, int size) {
  double result = 0;
  for (int i = 0; i < size; i++) {
    result += y[i] * kernel[i];
  }
  return result;
}

void kernel3_fused(double *H, double *dP, double *V, double *drdrx, double *R,
                   double *F2)
{
  cblas_dgemm_batch_strided(CblasRowMajor, CblasNoTrans, CblasNoTrans, c, d, k,
                            1, H, k, c * k, dP, d, k * d, 0, V, d, d * c,
                            a * b);
  int abc;
  double dM[3 * d];
  double rij, rijinv;
  for (int ia = 0; ia < a; ia++) {
    for (int ib = 0; ib < b; ib++) {
      for (int ic = 0; ic < c; ic++) {
        abc = ia * b * c + ib * c + ic;
        rij = R[abc];
        if (rij == 0) break;
        rijinv = 1.0 / rij;
        calculate_dM(m - 1, rijinv, drdrx + abc * 3, dM);
        double *V_ic = V + abc * d;
        for (int ix = 0; ix < 3; ix++) {
          F2[abc * 3 + ix] = simd_dot(V_ic, &dM[ix * d], d);
        }
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
  long long dM_size = a * b * c * x * d;

  R = (double *) malloc(R_size * sizeof(double));
  D = (double *) malloc(D_size * sizeof(double));
  dM = (double *) malloc(dM_size * sizeof(double));
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
      for (long long j = 0; j < 3 * d; j++) {
        dM[i * 3 * d + j] = 0;
      }
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
    double rijinv = 1.0 / rij;
    calculate_dM(3, rijinv, drdrx + i * 3, dM + i * 3 * d);
  }

  double *H;
  double *dP;
  double *V;
  double *F2;
  double *F2_fused;

  long long H_size = a * b * c * k;
  long long P_size = a * b * k * d;
  long long V_size = a * b * c * d;
  long long F_size = a * b * c * x;

  H = (double *)malloc(H_size * sizeof(double));
  dP = (double *)malloc(P_size * sizeof(double));
  V = (double *)malloc(V_size * sizeof(double));
  F2 = (double *)malloc(F_size * sizeof(double));
  F2_fused = (double *)malloc(F_size * sizeof(double));

  for (long long i = 0; i < F_size; i++) {
    F2[i] = 0.0;
    F2_fused[i] = 0.0;
  }
  for (long long i = 0; i < H_size; i++) {
    H[i] = randfrom(0.0, 1.0);
  }
  for (long long i = 0; i < P_size; i++) {
    dP[i] = randfrom(0.0, 1.0);
  }

  kernel3(H, dP, V, dM, F2);
  kernel3_fused(H, dP, V, drdrx, R, F2_fused);

  double diff = 0.0;
  for (long long i = 0; i < F_size; i++) {
    diff += fabs(F2[i] - F2_fused[i]);
  }

  if (world_rank == 0) {
    printf("Diff of kernel 3 and fused kernel 3: %f\n", diff);
  }

  double start, end;
  int iter = 5;

  MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();

  for (int i = 0; i < iter; i++) {
    kernel3(H, dP, V, dM, F2);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  end = MPI_Wtime();

  if (world_rank == 0) {
    printf("Kernel3 with %d processes computing time %f sec, ", world_size,
           (end - start) / iter);
    printf("using a=%d, b=%d, c=%d, d=%d, k=%d, m=%d, x=%d.\n", a, b, c, d, k,
           m - 1, x);
  }

  free(dM);
  free(H);
  free(dP);
  free(V);
  free(F2);
  free(R);
  free(D);
  free(drdrx);
  free(F2_fused);
  free(mask);

  MPI_Finalize();
  return 0;
}
