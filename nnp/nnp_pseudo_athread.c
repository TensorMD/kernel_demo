#include "nnp_pseudo_athread.h"
#include "cppblas.hpp"
#include "mpi.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static double simd_dot(double *y, double *kernel, int size)
{
  double result = 0;
  for (int i = 0; i < size; i++) { result += y[i] * kernel[i]; }
  return result;
}

static void simd_add(double *y, double *bias, int size)
{
  for (int i = 0; i < size; i++) { y[i] += bias[i]; }
}

static void activate(double *dy, double *y, int size)
{
  double z;
  for (int i = 0; i < size; i++) {
    z = sqrt(y[i] * y[i] + 4.0);
    y[i] = 0.5 * (y[i] + z);
    dy[i] = y[i] / z;
  }
}

void nnp_pseudo_athread(athread_config *config, int cid, MPI_Comm *comm)
{
  int block = config->block;
  int max_col = config->max_col;
  int *kernel_ngroups = config->kernel_ngroups;
  int *kernel_cpe_id = config->kernel_cpe_id;
  int *ncols_on_cpe = config->ncols_on_cpe;
  int *kernel_ngroups_bp = config->kernel_ngroups_bp;
  int *kernel_cpe_id_bp = config->kernel_cpe_id_bp;
  int *ncols_on_cpe_bp = config->ncols_on_cpe_bp;

  int M;
  int N;
  int max_moment;
  int beta;
  int nlayers;
  int max_layer;

  MPI_Bcast(&M, 1, MPI_INT, 64, MPI_COMM_WORLD);
  MPI_Bcast(&N, 1, MPI_INT, 64, MPI_COMM_WORLD);
  MPI_Bcast(&max_moment, 1, MPI_INT, 64, MPI_COMM_WORLD);
  MPI_Bcast(&beta, 1, MPI_INT, 64, MPI_COMM_WORLD);
  MPI_Bcast(&nlayers, 1, MPI_INT, 64, MPI_COMM_WORLD);
  MPI_Bcast(&max_layer, 1, MPI_INT, 64, MPI_COMM_WORLD);

  int *layer_sizes = (int *) malloc(nlayers * sizeof(int));
  int *input_sizes = (int *) malloc(nlayers * sizeof(int));

  MPI_Bcast(layer_sizes, nlayers, MPI_INT, 64, MPI_COMM_WORLD);
  MPI_Bcast(input_sizes, nlayers, MPI_INT, 64, MPI_COMM_WORLD);

  double *bias = (double *) malloc(nlayers * max_layer * sizeof(double));
  double *kernel_matrix_on_this_cpe_T =
      (double *) malloc(max_col * max_layer * sizeof(double));
  double *kernel_matrix_on_this_cpe_bp =
      (double *) malloc(max_col * max_layer * sizeof(double));
  double *last_kernel = (double *) malloc(max_layer * sizeof(double));

  MPI_Bcast(bias, nlayers * max_layer, MPI_DOUBLE, 64, MPI_COMM_WORLD);
  MPI_Bcast(last_kernel, max_layer, MPI_DOUBLE, 64, MPI_COMM_WORLD);
  MPI_Status status;
  MPI_Recv(kernel_matrix_on_this_cpe_T, max_col * max_layer, MPI_DOUBLE, 64, 4,
           MPI_COMM_WORLD, &status);
  MPI_Recv(kernel_matrix_on_this_cpe_bp, max_col * max_layer, MPI_DOUBLE, 64, 5,
           MPI_COMM_WORLD, &status);

  double *x = (double *) malloc(block * N * sizeof(double));
  double *y = (double *) malloc(block * N * sizeof(double));
  double *energy = (double *) malloc(block * sizeof(double));
  double *kernel = (double *) malloc(max_col * max_layer * sizeof(double));
  double *dy =
      (double *) malloc((nlayers - 1) * block * max_layer * sizeof(double));
  double *y_old = (double *) malloc(block * N * sizeof(double));
  double *y_temp = (double *) malloc(block * N * sizeof(double));

  for (int i = 0; i < M / (64 * block); i++) {
    MPI_Status status;
    MPI_Recv(x, block * N, MPI_DOUBLE, 64, 0, MPI_COMM_WORLD, &status);
    cppblas_copy(block * N, x, 1, y, 1);

    for (int j = 0; j < nlayers; j++) {
      if (j < nlayers - 1) {
        int num_groups = kernel_ngroups[j];
        int c0 = 0;

        for (int k = 0; k < num_groups; k++) {
          int target_cid = kernel_cpe_id[j * 16 + k];
          int ncols = ncols_on_cpe[j * 16 + k];
          if (cid == target_cid) {
            cppblas_copy(ncols * input_sizes[j], kernel_matrix_on_this_cpe_T, 1,
                         kernel, 1);
          }
          MPI_Bcast(kernel, ncols * input_sizes[j], MPI_DOUBLE, target_cid,
                    *comm);

          for (int batch = 0; batch < block; batch++) {
            for (int n = 0; n < ncols; n++) {
              y_temp[batch * N + c0 + n] =
                  cppblas_dot(input_sizes[j], y + batch * N, 1,
                              kernel + n * input_sizes[j], 1);
            }
          }
          c0 += ncols;
        }
        cppblas_copy(block * N, y_temp, 1, y, 1);

        for (int batch = 0; batch < block; batch++) {
          cppblas_axpy(layer_sizes[j], 1.0, bias + j * max_layer, 1,
                       y + batch * N, 1);
          activate(dy + j * block * max_layer + batch * max_layer,
                   y + batch * N, layer_sizes[j]);
        }

        if (j > 0 && layer_sizes[j - 1] == layer_sizes[j] && beta == 1) {
          for (int batch = 0; batch < block; batch++) {
            cppblas_axpy(layer_sizes[j], 1.0, y_old + batch * N, 1,
                         y + batch * N, 1);
          }
        }
        cppblas_copy(block * N, y, 1, y_old, 1);
      } else {
        for (int batch = 0; batch < block; batch++) {
          energy[batch] =
              cppblas_dot(input_sizes[j], y + batch * N, 1, last_kernel, 1) +
              bias[j * max_layer];
        }
      }
    }

    MPI_Send(energy, block, MPI_DOUBLE, 64, 1, MPI_COMM_WORLD);

    for (int j = nlayers - 2; j >= 0; j--) {
      if (beta == 1) {
        if (j == nlayers - 2) {
          for (int batch = 0; batch < block; batch++) {
            cppblas_copy(layer_sizes[j], last_kernel, 1, y_old + batch * N, 1);
          }
        } else {
          for (int batch = 0; batch < block; batch++) {
            cppblas_copy(layer_sizes[j], y + batch * N, 1, y_old + batch * N,
                         1);
          }
        }
      }

      if (j == nlayers - 2) {
        for (int batch = 0; batch < block; batch++) {
          for (int n = 0; n < layer_sizes[j]; n++) {
            dy[j * block * max_layer + batch * max_layer + n] *= last_kernel[n];
          }
        }
      } else {
        for (int batch = 0; batch < block; batch++) {
          for (int n = 0; n < layer_sizes[j]; n++) {
            dy[j * block * max_layer + batch * max_layer + n] *=
                y[batch * N + n];
          }
        }
      }

      int num_groups = kernel_ngroups_bp[j];
      int c0 = 0;

      for (int k = 0; k < num_groups; k++) {
        int target_cid = kernel_cpe_id_bp[j * 16 + k];
        int ncols = ncols_on_cpe_bp[j * 16 + k];
        if (cid == target_cid) {
          cppblas_copy(ncols * layer_sizes[j], kernel_matrix_on_this_cpe_bp, 1,
                       kernel, 1);
        }
        MPI_Bcast(kernel, ncols * layer_sizes[j], MPI_DOUBLE, target_cid,
                  *comm);

        for (int batch = 0; batch < block; batch++) {
          for (int n = 0; n < ncols; n++) {
            y[batch * N + c0 + n] = cppblas_dot(
                layer_sizes[j], dy + j * block * max_layer + batch * max_layer,
                1, kernel + n * layer_sizes[j], 1);
          }
        }
        c0 += ncols;
      }

      if (j > 0 && beta == 1 && layer_sizes[j] == layer_sizes[j - 1]) {
        for (int batch = 0; batch < block; batch++) {
          cppblas_axpy(layer_sizes[j], 1.0, y_old + batch * N, 1, y + batch * N,
                       1);
        }
      }
    }

    MPI_Send(y, block * N, MPI_DOUBLE, 64, 2, MPI_COMM_WORLD);

    for (int batch = 0; batch < block; batch++) {
      for (int pos = 0; pos < N; pos += max_moment + 1) {
        y[batch * N + pos] = 0.5 / x[batch * N + pos] * y[batch * N + pos];
      }
    }
  }
  free(layer_sizes);
  free(input_sizes);
  free(bias);
  free(kernel_matrix_on_this_cpe_T);
  free(kernel_matrix_on_this_cpe_bp);
  free(last_kernel);
  free(x);
  free(y);
  free(energy);
  free(kernel);
  free(dy);
  free(y_old);
  free(y_temp);
}