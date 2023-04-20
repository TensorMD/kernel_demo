#include "mkl.h"
#include "mpi.h"
#include "ration.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int main(int argc, char *argv[]) {
  int world_size, world_rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank == 0) { printf("Started %d processes.\n", world_size); }

  int a = 10000, b = 1, c = 200, d = 20, m = 4, x = 3;
  double rmax = 6.0, delta = 0.01;
  int algo = 1;
  int n = (int) (rmax / delta) + 1;
  int k[3] = {16, 32, 64};
  double r[64], q[64];
  double start, end;
  double *sij = new double[a * b * c];
  double *dsij = new double[a * b * c];
  double *fij = new double[a * b * c];
  double *dfij = new double[a * b * c];
  double *aij = new double[a * b * c];
  double *R = new double[a * b * c];

  srand(time(0));
  for (int i = 0; i < 64; i++) {
    r[i] = (double) rand() / RAND_MAX;
    q[i] = (double) rand() / RAND_MAX;
  }
  for (size_t i = 0; i < a * b * c; i++) {
    fij[i] = (double) rand() / RAND_MAX;
    dfij[i] = (double) rand() / RAND_MAX;
    aij[i] = (double) rand() / RAND_MAX;
    R[i] = (double) rand() / RAND_MAX;
  }

  for (int i = 0; i < 3; i++) {
    int n_out = k[i];
    double *y = new double[n_out * n];
    double *H = new double[a * b * c * n_out];
    double *dH = new double[a * b * c * n_out];

    srand(time(0));
    for (size_t i = 0; i < n_out * n; i++) {
      y[i] = (double) rand() / RAND_MAX;
    }

    BatchRation1d<double> *batch_ration = new BatchRation1d<double>();
    batch_ration->setup(n_out, n, delta, y);

    int test = 17;

    // Ration

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    batch_ration->compute(R, a * b * c, H, dH);
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    if (world_rank == 0) {
      printf("k = %d.\n  Ration: %f sec. H[%d] = %f.\n", n_out, (end - start), test, H[test]);
      fflush(stdout);
    }

    // Strided

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    // cosine cutoff
    mkl_domatcopy('R', 'N', 1, a * b * c, 1 / rmax, R, a * b * c, sij,
                  a * b * c);
    for (int j = 0; j < a * b * c; j++) {
      if (sij[j] > 1.0) sij[j] = 1.0;
    }
    cblas_dscal(a * b * c, M_PI, sij, 1);
    vdSin(a * b * c, sij, dsij);
    cblas_dscal(a * b * c, -0.5 * M_PI / rmax, dsij, 1);
    vdCos(a * b * c, sij, sij);
    cblas_dscal(a * b * c, 0.5, sij, 1);
    vdLinearFrac(a * b * c, sij, dsij, 1, 0.5, 0, 1, sij);
    for (int j = 0; j < n_out; j++) {
      // aij = x / r
      mkl_domatcopy('R', 'N', 1, a * b * c, 1 / r[j], R, a * b * c, aij,
                    a * b * c);
      // fij = exp(-(aij**q))
      vdPowx(a * b * c, aij, q[j], fij);
      cblas_dscal(a * b * c, -1, fij, 1);
      vdExp(a * b * c, fij, fij);
      // dfij = fij * aij**(q-1) * -q / r
      vdPowx(a * b * c, aij, q[j] - 1, dfij);
      vdMul(a * b * c, fij, dfij, dfij);
      cblas_dscal(a * b * c, -q[j] / r[j], dfij, 1);
      // H = fij * sij
      vdMulI(a * b * c, fij, 1, sij, 1, H + j, n_out);
      // dH = sij * dfij + dsij * fij
      vdMul(a * b * c, sij, dfij, dfij);
      vdMulI(a * b * c, dsij, 1, fij, 1, dH + j, n_out);
      vdAddI(a * b * c, dfij, 1, dH + j, n_out, dH + j, n_out);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    if (world_rank == 0) {
      printf("  Strided: %f sec. H[%d] = %f.\n", (end - start), test, H[test]);
      fflush(stdout);
    }

    // Transpose (Hkabc)

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    // cosine cutoff
    mkl_domatcopy('R', 'N', 1, a * b * c, 1 / rmax, R, a * b * c, sij,
                  a * b * c);
    for (int j = 0; j < a * b * c; j++) {
      if (sij[j] > 1.0) sij[j] = 1.0;
    }
    cblas_dscal(a * b * c, M_PI, sij, 1);
    vdSin(a * b * c, sij, dsij);
    cblas_dscal(a * b * c, -0.5 * M_PI / rmax, dsij, 1);
    vdCos(a * b * c, sij, sij);
    cblas_dscal(a * b * c, 0.5, sij, 1);
    vdLinearFrac(a * b * c, sij, dsij, 1, 0.5, 0, 1, sij);
    for (int j = 0; j < n_out; j++) {
      // aij = x / r
      mkl_domatcopy('R', 'N', 1, a * b * c, 1 / r[j], R, a * b * c, aij,
                    a * b * c);
      // fij = exp(-(aij**q))
      vdPowx(a * b * c, aij, q[j], fij);
      cblas_dscal(a * b * c, -1, fij, 1);
      vdExp(a * b * c, fij, fij);
      // dfij = fij * aij**(q-1) * -q / r
      vdPowx(a * b * c, aij, q[j] - 1, dfij);
      vdMul(a * b * c, fij, dfij, dfij);
      cblas_dscal(a * b * c, -q[j] / r[j], dfij, 1);
      // H = fij * sij
      vdMul(a * b * c, fij, sij, H + j * a * b * c);
      // dH = sij * dfij + dsij * fij
      vdMul(a * b * c, sij, dfij, dfij);
      vdMul(a * b * c, dsij, fij, dH + j * a * b * c);
      vdAdd(a * b * c, dfij, dH + j * a * b * c, dH + j * a * b * c);
    }
    mkl_dimatcopy('R', 'T', n_out, a * b * c, 1, H, a * b * c, n_out);
    mkl_dimatcopy('R', 'T', n_out, a * b * c, 1, dH, a * b * c, n_out);
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    if (world_rank == 0) {
      printf("  Transpose (Hkabc): %f sec. H[%d] = %f.\n", (end - start), test, H[test]);
      fflush(stdout);
    }

    // Naive

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    // cosine cutoff
    double coeff = 1 / rmax;
    for (int j = 0; j < a * b * c; j++){
      sij[j] = coeff * R[j];
      if (sij[j] > 1.0) sij[j] = 1.0;
      sij[j] = M_PI * sij[j];
      dsij[j] = -0.5 * M_PI * coeff * sin(sij[j]);
      sij[j] = 0.5 * cos(sij[j]) + 0.5;
    }
    for (int j = 0; j < n_out; j++) {
      for (int k = 0; k < a * b * c; k++) {
        // aij = x / r
        aij[k] = R[k] / r[j];
        // fij = exp(-(aij**q))
        fij[k] = exp(-pow(aij[k], q[j]));
        // dfij = fij * aij**(q-1) * -q / r
        dfij[k] = fij[k] * pow(aij[k], q[j] - 1) * -q[j] / r[j];
        // H = fij * sij
        H[k * n_out + j] = fij[k] * sij[k];
        // dH = sij * dfij + dsij * fij
        dH[k * n_out +j] = sij[k] * dfij[k] + dsij[k] * fij[k];
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    if (world_rank == 0) {
      printf("  Naive: %f sec. H[%d] = %f.\n", (end - start), test, H[test]);
    }

    delete batch_ration;
    delete[] y;
    delete[] H;
    delete[] dH;
  }
  delete[] sij;
  delete[] fij;
  delete[] dsij;
  delete[] dfij;
  delete[] aij;
  delete[] R;
  MPI_Finalize();
  return 0;
}