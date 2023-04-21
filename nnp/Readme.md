## dependency
- openblas or mkl(need to modify makefile)

## build
```shell
cd nnp
make
```

## usage
- mpirun -n 65 ./nnp.x
    - Verify the result of nnp_pseudo_athread
- mpirun -n 1 ./nnp.x
    - Run the NNP kernel of TensorMD

## notice
- a should be the multiple of 64 * block
- need to activate 65 mpi processes if verifing the nnp_pseudo_athread