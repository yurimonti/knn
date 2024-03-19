#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdbool.h>
#include <mpi.h>


typedef struct point_st {
    double x;
    double y;
    double z;
}t_point;
 

t_point* allocPoint() {
    t_point* point = (t_point*)malloc(sizeof(t_point));
    point->x = 0.0;
    point->y = 0.0;
    point->z = 0.0;
    return point;
}

void freePoint(t_point* point) {
    if (point != NULL) {
        point->x = 0.0;
        point->y = 0.0;
        point->z = 0.0;
    }
    free(point);
}

double euclideanDistance(t_point* point1, t_point* point2) {
    double x = point1->x - point2->x;
    double y = point1->y - point2->y;
    double z = point1->z - point2->z;
    return sqrt(x * x + y * y + z * z);
}

void generate_points(t_point* points, int num_points, double cube_length) {
    srand(time(0));
    for (int i = 0; i < num_points; i++) {
        points[i].x = (double)rand() / RAND_MAX * cube_length;
        points[i].y = (double)rand() / RAND_MAX * cube_length;
        points[i].z = (double)rand() / RAND_MAX * cube_length;
    }
}

int get_matrix_position(int row, int col, int n_col){
    int offset = row * n_col;
    return offset + col;
}

void right_shift_from_position(int *neigh, double *dist,int neigh_number,int from_pos,int point_idx){
    for (int r = neigh_number - 1; r > from_pos; r--){
        int current_pos = get_matrix_position(point_idx,r,neigh_number);
        int prev_pos = get_matrix_position(point_idx,r-1,neigh_number);
        dist[current_pos] = dist[prev_pos];
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

void fill_default_values(double *neigh_distance, int *neigh_idxes,int num_neigh,int num_points,int cube_dim){
    for (int i = 0; i < num_neigh*num_points; i++){
        neigh_distance[i] = cube_dim*sqrt(3) +1;
        neigh_idxes[i] = -1;
    }
}

void set_values_to_neigh(double *neigh_distances, int *neigh_idxes,int num_neigh,double distance,int point_idx,int from_pos,int neigh_idx){
    neigh_distances[get_matrix_position(point_idx, from_pos, num_neigh)] = distance;
    neigh_idxes[get_matrix_position(point_idx, from_pos, num_neigh)] = neigh_idx;
}

int main(int argc, char **argv){
    int my_rank,num_procs,N,K,cube_side_value,points_per_process;
    double start, finish;
    cube_side_value = 100;
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
    t_point *points = (t_point *) malloc(sizeof(t_point) * N);
    //? MPI start
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    /* find out number of processes */
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    //? creation of a structure for MPI --> point struct Type
    MPI_Datatype point_type;
    int block_length[] = {1,1,1};
    MPI_Aint displacements[] = {8,8,8};
    MPI_Datatype types[3] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
    MPI_Type_create_struct(3,block_length,displacements,types,&point_type);
    MPI_Type_commit(&point_type);
    
    
    //! generation
    // double *distance_matrix = (double *)malloc(sizeof(double)*((N*(N-1))/2));
    int *neighs_matrix;
    double *neigh_distances_matrix;
    if(my_rank == 0) {
        generate_points(points, N, cube_side_value);
        neighs_matrix = (int *)malloc(sizeof(int)*K*N);
        neigh_distances_matrix = (double *)malloc(sizeof(double)*K*N);
    }else{
        neighs_matrix = (int *)malloc(sizeof(int)*K*points_per_process);
        neigh_distances_matrix = (double *)malloc(sizeof(double)*K*points_per_process);
    }
    fill_default_values(neigh_distances_matrix,neighs_matrix,K,points_per_process,cube_side_value);
    //!
    MPI_Barrier(MPI_COMM_WORLD);
    //TIME
    start = MPI_Wtime();

    MPI_Bcast(points,N,point_type,0,MPI_COMM_WORLD);

    //COMPUTATION
    for (int i = points_per_process*my_rank; i < points_per_process*(my_rank+1); i++){
        for (int j = 0; j < N; j++){
            if(i == j) continue;
            //else {
                // write_value_matrix2(distance_matrix,i,j,N,euclideanDistance(&points[i],&points[j]));
                double dist = euclideanDistance(&points[i],&points[j]);
                //printf("euclidean distance between i:%d j:%d is: %f",i,j,dist);
                for (int h = 0; h < K; h++){
                    //anche per i neighbours
                    double neigh_dist = neigh_distances_matrix[get_matrix_position(i,h,K)];
                    if(dist < neigh_dist){
                        right_shift_from_position(neighs_matrix,neigh_distances_matrix,K,h,i);
                        set_values_to_neigh(neigh_distances_matrix,neighs_matrix,K,dist,i,h,j);
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
    // printf("\n================MIN===================\n");
    // for (int i = 0; i < N; i++)
    // {
    //     for (int j = 0; j < K; j++)
    //     {
    //         // double to_print = i==j ? 0.0 : read_value_matrix2(distance_matrix,i,j,N);
    //         printf("%f     ", neigh_distances_matrix[i * K + j]);
    //     }
    //     printf("\n");
    // }

    // printf("\n================NEIGH===================\n");
    // for (int i = 0; i < N; i++)
    // {
    //     for (int j = 0; j < K; j++)
    //     {
    //         // double to_print = i==j ? 0.0 : read_value_matrix2(distance_matrix,i,j,N);
    //         printf("%d     ", neighs_matrix[i * K + j]);
    //     }
    //     printf("\n");
    // }
    
    MPI_Gather(neigh_distances_matrix,K*points_per_process,MPI_DOUBLE,neigh_distances_matrix,K*points_per_process,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Gather(neighs_matrix,K*points_per_process,MPI_INT,neighs_matrix,K*points_per_process,MPI_INT,0,MPI_COMM_WORLD);

    finish = MPI_Wtime()-start;
    double max_time;
    MPI_Reduce(&finish,&max_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(my_rank ==0){
        printf("\n================MIN===================\n");
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < K; j++)
            {
                // double to_print = i==j ? 0.0 : read_value_matrix2(distance_matrix,i,j,N);
                printf("%f     ", neigh_distances_matrix[i * K + j]);
            }
            printf("\n");
        }
    }
    free(neigh_distances_matrix);
    free(neighs_matrix);
    // free(distance_matrix);
    free(points);
    MPI_Finalize();
    printf("Time elapsed: %fl seconds",max_time);
    return 0;
}
