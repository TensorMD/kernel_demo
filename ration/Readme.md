## dependency
- mkl

## build
```shell
cd ration
make
```

## usage
- mpirun -np <nprocs> ./main

## result
a = 10000
b = 1
c = 200

proccess = 1

| k = 16                           | k = 32                           | k = 64                           |
| ---------------------            | ------------------------------   | ----------------------           |
| Ration: 0.334017 sec.            | Ration: 0.649661 sec.            | Ration: 1.550356 sec.            |
| strided: 1.539242 sec.           | strided: 3.757301 sec.           | strided: 8.172815 sec.           |
| Hkabc then trans: 12.063281 sec. | Hkabc then trans: 17.166926 sec. | Hkabc then trans: 63.453690 sec. |

proccess = 5

| k = 16                           | k = 32                           | k = 64                           |
| ---------------------            | ------------------------------   | ----------------------           |
| Ration: 0.517204 sec.            | Ration: 0.955142 sec.            | Ration: 1.768079 sec.            |
| strided: 2.051783 sec.           | strided: 5.404943 sec.           | strided: 14.508618 sec.          |
| Hkabc then trans: 12.768596 sec. | Hkabc then trans: 18.067358 sec. | Hkabc then trans: 66.079249 sec. |

proccess = 40

| k = 16                           | k = 32                           | k = 64                           |
| ---------------------            | ------------------------------   | ----------------------           |
| Ration: 2.405599 sec.            | Ration: 2.550248 sec.            | Ration: 5.459778 sec.           |
| strided: 11.847195 sec.          | strided: 41.640327 sec.          | strided: 165.914218 sec.         |
| Hkabc then trans: 18.336876 sec. | Hkabc then trans: 29.945404 sec. | Hkabc then trans: 104.560517 sec.|
