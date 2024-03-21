#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdbool.h>

typedef struct point_st {
    double x;
    double y;
    double z;
}t_point;

double euclideanDistance(t_point* point1, t_point* point2) {
    double x = point1->x - point2->x;
    double y = point1->y - point2->y;
    double z = point1->z - point2->z;
    return sqrt(x * x + y * y + z * z);
}

void generate_points(t_point* points, int num_points, double cube_length) {
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

// void right_shift_from_position(int *neigh, double *dist,int neigh_number,int from_pos,int point_idx){
//     for (int r = neigh_number - 1; r > from_pos; r--){
//         int current_pos = get_matrix_position(point_idx,r,neigh_number);
//         int prev_pos = get_matrix_position(point_idx,r-1,neigh_number);
//         dist[current_pos] = dist[prev_pos];
//         neigh[current_pos] = neigh[prev_pos];
//     }
// }

int find_position(double *array, int left, int right, double to_insert)
{
    while (left <= right)
    {
        int mid = (left + right) / 2;

        if (array[mid] == to_insert)
        {
            return mid;
        }
        else if (array[mid] < to_insert)
        {
            left = mid + 1;
        }
        else
        {
            right = mid - 1;
        }
    }

    return left;
}

void insert_value(double *array, int *array2, int array_dim, double distance_to_insert,int neigh_to_insert,int idx_point)
{
    int position = find_position(array, idx_point*array_dim , (idx_point+1)*array_dim - 1, distance_to_insert);
    for (int i = (idx_point+1)*array_dim - 1; i > position; i--)
    {
        array[i] = array[i - 1];
        array2[i] = array2[i - 1];
    }
    array[position] = distance_to_insert;
    array2[position] = neigh_to_insert;
    //printf("position changed %lf\n",array[position]);
    //(array_dim)++;
}

void print_error_argc(int argc){
    fprintf(stderr, "\n\nWrong number of parameters! Expected 3, but are %d!\n", argc);
    fprintf(stderr, "\n\nValid parameters are knnmpi {number_of_points} {number_of_neighbours}\n");
}

void print_error_neighbours(int points_number, int neighbours_number){
    fprintf(stderr, "\n\nNeighbours to find are more than the total number of points! Expected < %d, but are %d!\n",points_number,neighbours_number);
}

int get_offset(int row , int col, int matrix_dim){
    if(row < col){
        int c = 0;
        for (int i = 0; i < row; i++) c -=i;
        int pos = c + row * (matrix_dim -1)+ col -1;
        return pos;
    } if (row > col) return get_offset(col,row,matrix_dim);
}

double read_value_matrix(double *matrix,int row,int col,int n_cols){
    int offset = get_offset(row,col,n_cols);
    return matrix[offset];
}

void write_value_matrix(double *matrix,int row,int col,int n_cols, double to_set){
    int offset = get_offset(row,col,n_cols);
    matrix[offset] = to_set;
}

double dwalltime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double sec = tv.tv_sec + tv.tv_usec / 1000000.0;
    return sec;
}

void fill_default_values(double *neigh_distance, int *neigh_idxes,int num_neigh,int num_points){
    for (int i = 0; i < num_neigh*num_points; i++){
        neigh_distance[i] = 100*sqrt(3) +1;
        neigh_idxes[i] = -1;
    }
}

// void set_values_to_neigh(double *neigh_distances, int *neigh_idxes,int num_neigh,double distance,int point_idx,int from_pos,int neigh_idx){
//     neigh_distances[get_matrix_position(point_idx, from_pos, num_neigh)] = distance;
//     neigh_idxes[get_matrix_position(point_idx, from_pos, num_neigh)] = neigh_idx;
// }

int main(int argc, char **argv){
    if(argc != 3) {
        print_error_argc(argc);
        return -1;
    }
    int N = atoi(argv[1]);
    int K = atoi(argv[2]);
    if(K>=N){
        print_error_neighbours(N,K);
        return -1;
    }
    //SETTING UP
    t_point *points = (t_point *) malloc(sizeof(t_point) * N);
    //double *distance_matrix = (double *)malloc(sizeof(double)*((N*(N-1))/2));
    int *neighs_matrix = (int *)malloc(sizeof(int)*K*N);
    double *neigh_distances_matrix = (double *)malloc(sizeof(double)*K*N);
    srand(time(0));
    generate_points(points, N, 100);
    fill_default_values(neigh_distances_matrix,neighs_matrix,K,N);

    //TIME
    double tick = dwalltime();

    //COMPUTATION
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            if(i == j) continue;
            //else {
                //write_value_matrix(distance_matrix,i,j,N,euclideanDistance(&points[i],&points[j]));
                double dist = euclideanDistance(&points[i],&points[j]);
                //printf("euclidean distance between i:%d j:%d is: %f",i,j,dist);
                //TODO: substitute with new functions
                insert_value(neigh_distances_matrix,neighs_matrix,K,dist,j,i);
                // for (int h = 0; h < K; h++){
                //     //anche per i neighbours
                //     double neigh_dist = neigh_distances_matrix[get_matrix_position(i,h,K)];
                //     if(dist < neigh_dist){
                //         right_shift_from_position(neighs_matrix,neigh_distances_matrix,K,h,i);
                //         set_values_to_neigh(neigh_distances_matrix,neighs_matrix,K,dist,i,h,j);
                //         break;
                //     }
                // }
                
            //}
        }
    }
    printf("Total time=%lf\n", dwalltime() - tick);
    // printf("\n=============COMPLETE MATRIX======================\n");
    // for (int i = 0; i < N; i++){
    //     for (int j = 0; j < N; j++){
    //         double to_print = i==j ? 0.0 : read_value_matrix(distance_matrix,i,j,N);
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

    free(neigh_distances_matrix);
    free(neighs_matrix);
    //free(distance_matrix);
    free(points);
return EXIT_SUCCESS;
}