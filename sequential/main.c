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

void right_shift_from_position(int *neigh, double *dist,int neigh_number,int from_pos,int point_idx){
    for (int r = neigh_number - 1; r > from_pos; r--){
        dist[point_idx * neigh_number + r] = dist[point_idx * neigh_number + (r - 1)];
        neigh[point_idx * neigh_number + r] = neigh[point_idx * neigh_number + (r - 1)];
    }
}

void print_error_argc(int argc){
    fprintf(stderr, "\n\nWrong number of parameters! Expected 3, but are %d!\n", argc);
    fprintf(stderr, "\n\nValid parameters are knnmpi {number_of_points} {number_of_neighbours}\n");
}

void print_error_neighbours(int points_number, int neighbours_number){
    fprintf(stderr, "\n\nNeighbours to find are more than the total number of points! Expected < %d, but are %d!\n",points_number,neighbours_number);
}

// int findPosition(double *min_distances, double distance, int num_neigh) {
//     int left_limit, right_limit, middle;
//     bool found;

//     left_limit = 0;
//     right_limit = num_neigh - 1;
//     middle = (int) (left_limit + right_limit)/2;
//     found = false;

//     // Binary search for position
//     while((left_limit < right_limit) && !found) {
//         if (min_distances[middle] < distance) {
//             left_limit = middle + 1;
//             middle = (middle + right_limit) / 2;
//         }
//         else if (min_distances[middle] > distance) {
//             right_limit = middle - 1;
//             if(middle ==0 && left_limit == 0) middle = 0;
//             else middle = (middle + left_limit) / 2;
//         }
//         else
//             found = true;
//     }

//     return middle;
// }

// void insertDistance(double *min_distances, double distance,int num_neigh, int pos) {
//     //Update min distances and neighbours
//     int i;
//     for (i = num_neigh - 1; i > pos; i--) {
//         min_distances[i] = min_distances[i-1];
//     }
//     min_distances[pos] = distance;
// }

// void updateNeighbours(double *min_distances, int num_neigh, double *distances, int num_points){
//     int i, j,pos;
//     double distance;

//     for (i=0; i < num_points; i++) {

//         distance = distances[i];

//         pos = findPosition(&min_distances[i], distance, num_neigh);

//         while (pos < (num_neigh - 1) && min_distances[(i + pos) % num_neigh] <= distance)
//             pos++;

//         if (min_distances[(i + pos) % num_neigh] > distance){
//             insertDistance(&min_distances[i], distance, num_neigh, pos);
//         }
//     }
// }
// void calculate_distance(t_point *points,int num_point,double *distances){
//     for(int i = 0; i< num_point;i++){
//         distancespoints[i]
//     }
// }

// void alloc_matrix(double *matrix_to_alloc, int rows,int cols){
//     matrix_to_alloc = (double *) malloc(rows * cols * sizeof(double));
// }

// double read_value_matrix(double *matrix,int row,int col,int n_cols){
//     int offset = row * n_cols + col;
//     return matrix[offset];
// }

// void write_value_matrix(double *matrix,int row,int col,int n_cols, double to_set){
//     int offset = row * n_cols + col;
//     matrix[offset] = to_set;
// }

// void alloc_matrix2(double *matrix_to_alloc, int rows,int cols){
//     matrix_to_alloc = (double *) malloc(rows * cols * sizeof(double));
// }

int get_offset(int row , int col, int matrix_dim){
    if(row < col){
        int c = 0;
        for (int i = 0; i < row; i++) c -=i;
        int pos = c + row * (matrix_dim -1)+ col -1;
        return pos;
    } if (row > col) return get_offset(col,row,matrix_dim);
}

double read_value_matrix2(double *matrix,int row,int col,int n_cols){
    int offset = get_offset(row,col,n_cols);
    return matrix[offset];
}

void write_value_matrix2(double *matrix,int row,int col,int n_cols, double to_set){
    int offset = get_offset(row,col,n_cols);
    matrix[offset] = to_set;
}

double dwalltime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double sec = tv.tv_sec + tv.tv_usec / 1000000.0;

    return sec;
}

int main(int argc, char **argv)
{
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
    double *distance_matrix = (double *)malloc(sizeof(double)*((N*(N-1))/2));
    int *neighs_matrix = (int *)malloc(sizeof(int)*K*N);
    double *neigh_distances_matrix = (double *)malloc(sizeof(double)*K*N);
    generate_points(points, N, 100);
    for (int i = 0; i < K*N; i++){
        neigh_distances_matrix[i] = 100*sqrt(3) +1;
        neighs_matrix[i] = -1;
    }

    //TIME
    double tick = dwalltime();

    //COMPUTATION
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            if(i == j) continue;
            //else {
                //write_value_matrix2(distance_matrix,i,j,N,euclideanDistance(&points[i],&points[j]));
                double dist = euclideanDistance(&points[i],&points[j]);
                //printf("euclidean distance between i:%d j:%d is: %f",i,j,dist);
                for (int h = 0; h < K; h++){
                    //anche per i neighbours
                    double neigh_dist = neigh_distances_matrix[i*K + h];
                    if(dist< neigh_dist){
                        right_shift_from_position(neighs_matrix,neigh_distances_matrix,K,h,i);
                        neigh_distances_matrix[i * K + h] = dist;
                        neighs_matrix[i * K + h] = j;
                        break;
                    }
                }
                
            //}
        }
    }
    printf("Total time=%lf\n", dwalltime() - tick);
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

    free(neigh_distances_matrix);
    free(neighs_matrix);
    free(distance_matrix);
    free(points);
    // double array[3] = {7,3,5};
    // double *to_set = (double *)malloc(3 * sizeof(double));
    // updateNeighbours(to_set,3,array,3); 
    // printf("[%f,%f,%f]\n",to_set[0],to_set[1],to_set[2]);
    // free(to_set);
    // t_point *points = (t_point *)malloc(N * sizeof(t_point));
    // generate_points(points, N, 100);
    // double *matrix_to_alloc = (double *) malloc(N * N * sizeof(double));
    // //calculate_distance(&points);
    // for(int i = 0; i<N-1; i++){
    //     for(int j = i+1; j<N; j++){
    //         write_value_matrix(matrix_to_alloc,i,j,N,euclideanDistance(&points[i],&points[j]));
    //     }
    // }

    // for (int j = 0; j < N; j++){
    //         printf("(%lf,%lf,%lf)     ", points[j].x,points[j].y,points[j].z);
    // }
    // printf("\n===================================\n");
    // for (int i = 0; i < N; i++){
    //     for (int j = 0; j < N; j++){
    //         printf("%f     ", read_value_matrix(matrix_to_alloc,i,j,N));
    //     }
    //     printf("\n");
    // }
    // // for(int j = 0; j<N; j++){
    // //     double to_print = read_value_matrix(matrix_to_alloc,0,j,N);
    // //     printf("%lf ",to_print);
    // // }
    // printf("\n");
    // free(points);
    // free(matrix_to_alloc);
return EXIT_SUCCESS;
}
