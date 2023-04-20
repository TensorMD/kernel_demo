#include "cnpy.h"
#include "mpi.h"
#include "nnp.h"
#include "nnp_pseudo_athread.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static size_t a = 25600, b = 1, k = 128, m = 4;
static int block = 4;
static int kernel_ngroups[4] = {16, 4, 4, 4};
static int kernel_cpe_id[4][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {16, 17, 18, 19},
    {20, 21, 22, 23},
    {24, 25, 26, 27}};
static int ncols_on_cpe[4][16] = {
    {8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8},
    {32, 32, 32, 32},
    {32, 32, 32, 32},
    {32, 32, 32, 32}};
static int kernel_ngroups_bp[4] = {16, 4, 4, 4};
static int kernel_cpe_id_bp[4][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {16, 17, 18, 19},
    {20, 21, 22, 23},
    {24, 25, 26, 27}};
static int ncols_on_cpe_bp[4][16] = {
    {32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32},
    {32, 32, 32, 32},
    {32, 32, 32, 32},
    {32, 32, 32, 32}};
static int max_col = 32;

int main(int argc, char *argv[])
{
  int world_size, world_rank;
  MPI_Comm comm_worker;
  MPI_Group group_world, group_worker;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  if (world_rank == 0 && world_size != 65 && world_size != 1) {
    printf("MPI processes must be 65 or 1.\n");
    return 1;
  }

  if (world_size == 65) {
    const int excl[1] = {64};
    MPI_Comm_group(MPI_COMM_WORLD, &group_world);
    MPI_Group_excl(group_world, 1, excl, &group_worker);
    MPI_Comm_create(MPI_COMM_WORLD, group_worker, &comm_worker);
  }

  if (world_size == 65 && world_rank >= 0 && world_rank <= 63) {
    athread_config config;
    config.block = block;
    config.max_col = max_col;
    config.kernel_ngroups = kernel_ngroups;
    config.kernel_cpe_id = &kernel_cpe_id[0][0];
    config.ncols_on_cpe = &ncols_on_cpe[0][0];
    config.kernel_ngroups_bp = kernel_ngroups_bp;
    config.kernel_cpe_id_bp = &kernel_cpe_id_bp[0][0];
    config.ncols_on_cpe_bp = &ncols_on_cpe_bp[0][0];

    nnp_pseudo_athread(&config, world_rank, &comm_worker);
  }

  if ((world_rank == 64 && world_size == 65) ||
      (world_rank == 0 && world_size == 1)) {
    auto model = new NeuralNetworkPotential<double>();
    auto npz = cnpy::npz_load("Mo_fp64_large.npz");
    model->setup_global(npz);

    size_t G_size = a * b * k * m;
    double *G = new double[G_size];
    double *y = new double[a];
    double *dydx = new double[G_size];

    srand(time(0));
    for (size_t i = 0; i < G_size; i++) { G[i] = rand() / RAND_MAX; }

    model->compute(0, G, a, y, dydx);

    if (world_size == 65) {
      int M = a;
      int N = b * k * m;
      int max_moment = m - 1;
      int beta;
      int nlayers;
      int max_layer;

      MPI_Bcast(&M, 1, MPI_INT, 64, MPI_COMM_WORLD);
      MPI_Bcast(&N, 1, MPI_INT, 64, MPI_COMM_WORLD);
      MPI_Bcast(&max_moment, 1, MPI_INT, 64, MPI_COMM_WORLD);
      model->get_config(&beta, &nlayers, &max_layer);
      MPI_Bcast(&beta, 1, MPI_INT, 64, MPI_COMM_WORLD);
      MPI_Bcast(&nlayers, 1, MPI_INT, 64, MPI_COMM_WORLD);
      MPI_Bcast(&max_layer, 1, MPI_INT, 64, MPI_COMM_WORLD);

      int *layer_sizes = new int[nlayers];
      int *input_sizes = new int[nlayers];

      model->get_layer_sizes(layer_sizes, input_sizes);
      MPI_Bcast(layer_sizes, nlayers, MPI_INT, 64, MPI_COMM_WORLD);
      MPI_Bcast(input_sizes, nlayers, MPI_INT, 64, MPI_COMM_WORLD);

      double *bias = new double[nlayers * max_layer];
      double *kernel_matrix_on_this_cpe_T = new double[max_col * max_layer];
      double *kernel_matrix_on_this_cpe_bp = new double[max_col * max_layer];
      double *last_kernel = new double[max_layer];

      model->get_bias(bias);
      model->get_last_kernel(last_kernel);
      MPI_Bcast(bias, nlayers * max_layer, MPI_DOUBLE, 64, MPI_COMM_WORLD);
      MPI_Bcast(last_kernel, max_layer, MPI_DOUBLE, 64, MPI_COMM_WORLD);

      for (int cid = 0; cid < 64; cid++) {
        model->get_parameter(kernel_matrix_on_this_cpe_T,
                             kernel_matrix_on_this_cpe_bp, cid);
        MPI_Send(kernel_matrix_on_this_cpe_T, max_col * max_layer, MPI_DOUBLE,
                 cid, 4, MPI_COMM_WORLD);
        MPI_Send(kernel_matrix_on_this_cpe_bp, max_col * max_layer, MPI_DOUBLE,
                 cid, 5, MPI_COMM_WORLD);
      }

      double *y_pseudo_athread = new double[a];
      double *dydx_pseudo_athread = new double[G_size];

      MPI_Status status;
      for (int i = 0; i < a / (64 * block); i++) {
        for (int cid = 0; cid < 64; cid++) {
          int row = (i * 64 + cid) * block;
          MPI_Send(G + row * b * k * m, block * b * k * m, MPI_DOUBLE, cid, 0,
                   MPI_COMM_WORLD);
        }
        for (int cid = 0; cid < 64; cid++) {
          int row = (i * 64 + cid) * block;
          MPI_Recv(y_pseudo_athread + row, block, MPI_DOUBLE, cid, 1,
                   MPI_COMM_WORLD, &status);
        }
        for (int cid = 0; cid < 64; cid++) {
          int row = (i * 64 + cid) * block;
          MPI_Recv(dydx_pseudo_athread + row * b * k * m, block * b * k * m,
                   MPI_DOUBLE, cid, 2, MPI_COMM_WORLD, &status);
        }
      }

      bool y_flag = true;
      bool dydx_flag = true;
      for (int i = 0; i < a; i++) {
        if (abs(y_pseudo_athread[i] - y[i]) > 10e-6) {
          y_flag = false;
          break;
        }
      }
      for (int i = 0; i < G_size; i++) {
        if (abs(dydx_pseudo_athread[i] - dydx[i]) > 10e-6) {
          dydx_flag = false;
          break;
        }
      }

      if (y_flag) {
        printf("y test passed.\n");
      } else {
        printf("y test failed.\n");
      }
      if (dydx_flag) {
        printf("dydx test passed.\n");
      } else {
        printf("dydx test failed.\n");
      }

      delete[] y_pseudo_athread;
      delete[] dydx_pseudo_athread;
      delete[] bias;
      delete[] kernel_matrix_on_this_cpe_T;
      delete[] kernel_matrix_on_this_cpe_bp;
      delete[] last_kernel;
    }

    delete[] G;
    delete[] y;
    delete[] dydx;
    delete model;
  }

  MPI_Finalize();
  return 0;
}
