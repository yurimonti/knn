#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdbool.h>
#include <mpi.h>


double difference(int num1,int num2) {
    return num1>=num2 ? num1-num2 : num2-num1;
}

void generate_numbers(int* numbers, int num_numbers, int cube_length) {
    //srand(time(0));
    for (int i = 0; i < num_numbers; i++) {
        numbers[i] = (rand() % cube_length) +1;
    }
}

int get_matrix_position(int col, int row, int n_row){
    int offset = col * n_row;
    return offset + row;
}

void right_shift_from_position(int *neigh, int *diffs,int neigh_number,int from_pos,int point_idx){
    for (int r = neigh_number - 1; r > from_pos; r--){
        int current_pos = get_matrix_position(point_idx,r,neigh_number);
        int prev_pos = get_matrix_position(point_idx,r-1,neigh_number);
        diffs[current_pos] = diffs[prev_pos];
        neigh[current_pos] = neigh[prev_pos];
    }
}

void print_error_argc(int argc){
    fprintf(stderr, "\n\nWrong number of parameters! Expected 3, but are %d!\n", argc);
    fprintf(stderr, "\n\nValid parameters are knnmpi {number_of_points} {number_of_neighbours}\n");
}

void print_error_neighbours(int points_number, int neighbours_number){
    fprintf(stderr, "\n\nNeighbours to find are more than the total number of points! Expected < %d, but are %d!\n",points_number,neighbours_number);
}

// int get_offset(int row , int col, int matrix_dim){
//     if(row < col){
//         int c = 0;
//         for (int i = 0; i < row; i++) c -=i;
//         int pos = c + row * (matrix_dim -1)+ col -1;
//         return pos;
//     } if (row > col) return get_offset(col,row,matrix_dim);
// }

// double read_value_matrix2(double *matrix,int row,int col,int n_cols){
//     int offset = get_offset(row,col,n_cols);
//     return matrix[offset];
// }

// void write_value_matrix2(double *matrix,int row,int col,int n_cols, double to_set){
//     int offset = get_offset(row,col,n_cols);
//     matrix[offset] = to_set;
// }

void fill_default_values(int *neigh_distance, int *neigh_idxes,int num_neigh,int num_points,int cube_dim){
    for (int i = 0; i < num_neigh*num_points; i++){
        neigh_distance[i] = cube_dim;
        neigh_idxes[i] = -1;
    }
}

void set_values_to_neigh(int *neigh_distances, int *neigh_idxes,int num_neigh,int distance,int point_idx,int from_pos,int neigh_idx){
    neigh_distances[get_matrix_position(point_idx, from_pos, num_neigh)] = distance;
    neigh_idxes[get_matrix_position(point_idx, from_pos, num_neigh)] = neigh_idx;
}

int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    int my_rank,num_procs,N,K;
    int cube_side_value,points_per_process;
    int *neighs_matrix;
    int *neigh_distances_matrix;
    int *r_buffer_distances;
    int *r_buffer_neighs;
    int *numbers;
    // double start, finish;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    // cube_side_value = 100;
    if(argc != 3) {
        print_error_argc(argc);
        return -1;
    }
    N = atoi(argv[1]);
    K = atoi(argv[2]);
    if(K>=N){
        print_error_neighbours(N,K);
        return -1;
    }
    points_per_process = N / num_procs;
    //SETTING UP
    numbers = (int *) malloc(sizeof(int) * N);
    cube_side_value = 100;
    // printf("points x process : %d\n",points_per_process);
    // printf("points x process : %d\n",num_procs);
    //? MPI start
    // MPI_Init(&argc, &argv);
    // MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    // /* find out number of processes */
    // MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    //? creation of a structure for MPI --> point struct Type
    // MPI_Datatype point_type;
    // int block_length[] = {1,1,1};
    // MPI_Aint displacements[] = {8,8,8};
    // MPI_Datatype types[3] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
    // MPI_Type_create_struct(3,block_length,displacements,types,&point_type);
    // MPI_Type_commit(&point_type);
    
    
    // double *distance_matrix = (double *)malloc(sizeof(double)*((N*(N-1))/2));
    // int *neighs_matrix;
    // double *neigh_distances_matrix;
    // double *r_buffer_distances;
    // int *r_buffer_neighs;

    if(my_rank == 0) {
        srand(time(0));
        generate_numbers(numbers, N, cube_side_value);
        r_buffer_neighs = (int *)malloc(sizeof(int)*K*N);
        r_buffer_distances = (int *)malloc(sizeof(int)*K*N);
    }
    neighs_matrix = (int *)malloc(sizeof(int)*K*points_per_process);
    neigh_distances_matrix = (int *)malloc(sizeof(int)*K*points_per_process);
    fill_default_values(neigh_distances_matrix,neighs_matrix,K,points_per_process,cube_side_value);


    MPI_Barrier(MPI_COMM_WORLD);
    //TIME
    //start = MPI_Wtime();

    MPI_Bcast(numbers,N,MPI_INT,0,MPI_COMM_WORLD);

    //COMPUTATION
    for (int i = points_per_process*my_rank; i < points_per_process*(my_rank+1); i++){
        for (int j =  0; j < N; j++){
            if(i == j) continue;
            //else {
                // write_value_matrix2(distance_matrix,i,j,N,euclideanDistance(&points[i],&points[j]));
            int dist = difference(numbers[i],numbers[j]);
            //printf("euclidean distance between i:%d j:%d is: %f",i,j,dist);
            for (int h = 0; h < K; h++){
                int neigh_dist = neigh_distances_matrix[((i % points_per_process) * K) + h];
                if(dist < neigh_dist){
                    right_shift_from_position(neighs_matrix,neigh_distances_matrix,K,h,(i % points_per_process));
                    set_values_to_neigh(neigh_distances_matrix,neighs_matrix,K,dist,(i % points_per_process),h,j);
                    break;
                }
            }
            //}
        }
    }
    // printf("\n=============COMPLETE MATRIX======================\n");
    // for (int i = 0; i < N; i++){
    //     for (int j = 0; j < N; j++){
    //         double to_print = i==j ? 0.0 : read_value_matrix2(distance_matrix,i,j,N);
    //         printf("%f     ", to_print);
    //     }
    //     printf("\n");
    // }
    // 
    
    MPI_Gather(neigh_distances_matrix,K*points_per_process,MPI_INT,r_buffer_distances,K*points_per_process,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Gather(neighs_matrix,K*points_per_process,MPI_INT,r_buffer_neighs,K*points_per_process,MPI_INT,0,MPI_COMM_WORLD);


    if(my_rank ==0){
        printf("\n================NUMBERS===================\n");
        for (int j = 0; j < N; j++)
        {
            // double to_print = i==j ? 0.0 : read_value_matrix2(distance_matrix,i,j,N);
            printf("%d      ",numbers[j]);
        }   
        printf("\n");
    }

    printf("\n================MIN RANK = %d===================\n",my_rank);
    for (int i = 0; i < points_per_process; i++)
    {
        for (int j = 0; j < K; j++)
        {
            // double to_print = i==j ? 0.0 : read_value_matrix2(distance_matrix,i,j,N);
            printf("%d     ", neigh_distances_matrix[i * K + j]);
        }
        printf("\n");
    }

    printf("\n================NEIGH RANK = %d===================\n",my_rank);
    for (int i = 0; i < points_per_process; i++)
    {
        for (int j = 0; j < K; j++)
        {
            // double to_print = i==j ? 0.0 : read_value_matrix2(distance_matrix,i,j,N);
            printf("%d     ", neighs_matrix[i * K + j]);
        }
        printf("\n");
    }
    //finish = MPI_Wtime()-start;
    //double max_time;
    // MPI_Reduce(&finish,&max_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(my_rank ==0){
        
        printf("\n================MIN TOTAL===================\n");
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < K; j++)
            {
                // double to_print = i==j ? 0.0 : read_value_matrix2(distance_matrix,i,j,N);
                printf("%d     ", r_buffer_distances[i*K+j]);
            }
            printf("\n");
        }
        printf("\n================NEIGH TOTAL===================\n");
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < K; j++)
            {
                // double to_print = i==j ? 0.0 : read_value_matrix2(distance_matrix,i,j,N);
                printf("%d     ", r_buffer_neighs[i*K+j]);
            }
            printf("\n");
        }
    }
    if(my_rank ==0){
        free(r_buffer_distances);
        free(r_buffer_neighs);
    }
    free(neigh_distances_matrix);
    free(neighs_matrix);
    free(numbers);
    MPI_Finalize();
    //printf("Time elapsed: %fl seconds",max_time);
    return EXIT_SUCCESS;
}
