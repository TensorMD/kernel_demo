#include "mpi.h"

struct athread_config {
  int block;
  int max_col;
  int *kernel_ngroups;
  int *kernel_cpe_id;
  int *ncols_on_cpe;
  int *kernel_ngroups_bp;
  int *kernel_cpe_id_bp;
  int *ncols_on_cpe_bp;
};

extern "C" {
void nnp_pseudo_athread(athread_config *config, int cid, MPI_Comm *comm);
}