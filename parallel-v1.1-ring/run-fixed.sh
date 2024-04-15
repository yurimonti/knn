N=( 1024 2048 4096 8192 16384 32768 65536 )
P=( 2 4 8 16 32 )
K=( 5 10 15 20 )

mpicc -o knnmpi1-1-ring main.c -lm

for k in "${K[@]}"
do 
    for n in "${N[@]}"
    do
        for p in "${P[@]}"
        do
            mpirun -np $p ./knnmpi1-1-ring $n $k load.txt
        done
    done
done