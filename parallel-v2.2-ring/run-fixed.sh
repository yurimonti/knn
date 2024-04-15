N=( 1024 2048 4096 8192 16384 32768 65536 )
P=( 2 4 8 16 32 )
K=( 5 10 15 20 )

mpicc -o knnmpi2-2-ring main.c -lm

for k in "${K[@]}"
do 
    for n in "${N[@]}"
    do
        for p in "${P[@]}"
        do
            mpirun -np $p ./knnmpi2-2-ring $n $k load.txt
        done
    done
done