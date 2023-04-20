a_array=(5000 10000 30000 50000 100000 150000 200000)
k_array=(64 96 128)

np=1

for k in ${k_array[*]}
do
    for a in ${a_array[*]}
    do
        mpirun -np $np ./kernel1 -a$a -k$k
        mpirun -np $np ./kernel2 -a$a -k$k
        mpirun -np $np ./kernel3 -a$a -k$k
        mpirun -np $np ./kernel4 -a$a -k$k
    done
done