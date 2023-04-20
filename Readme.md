## dependency
- Intel oneAPI

## build
```shell
cd tensoralloy_demo
make
```

## usage
- mpirun -np <nprocs> ./kernel1 [-a<value_a>] [-b<value_b>] [-c<value_c>] [-k<value_k>] [-m<value_m>]
    - value_a   &nbsp;&nbsp;&nbsp;1-100000
    - value_b   &nbsp;&nbsp;&nbsp;1-5
    - value_c   &nbsp;&nbsp;&nbsp;1-500
    - value_k   &nbsp;&nbsp;&nbsp;8-128
    - value_m   &nbsp;&nbsp;&nbsp;0-3
    
