N=( 1024 2048 4096 8192 16384 32768 65536 )
K=( 5 10 15 20 )

gcc -o knnseq2-0 main.c -lm

for k in "${K[@]}"
do 
    for n in "${N[@]}"
    do
        ./knnseq2-0 $n $k 
    done
done