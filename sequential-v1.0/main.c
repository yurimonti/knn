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

void generate_points(t_point* points, int num_points, int cube_length) {
    int i;
    for (i = 0; i < num_points; i++) {
        points[i].x = (double)rand() / RAND_MAX * cube_length;
        points[i].y = (double)rand() / RAND_MAX * cube_length;
        points[i].z = (double)rand() / RAND_MAX * cube_length;
    }
}

double actual_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double sec = tv.tv_sec + tv.tv_usec / 1000000.0;
    return sec;
}

int get_matrix_position(int col, int row, int n_row){
    int offset = col * n_row;
    return offset + row;
}

void right_shift_from_position(int *neigh, double *dist,int neigh_number,int from_pos,int point_idx){
    int r;
    for (r = neigh_number - 1; r > from_pos; r--){
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

void fill_default_values(double *neigh_distance, int *neigh_idxes,int num_neigh,int num_points,int cube_dim){
    int i;
    for (i = 0; i < num_neigh*num_points; i++){
        neigh_distance[i] = cube_dim*sqrt(3) +1;
        neigh_idxes[i] = -1;
    }
}

void set_values_to_neigh(double *neigh_distances, int *neigh_idxes,int num_neigh,double distance,int point_idx,int from_pos,int neigh_idx){
    neigh_distances[get_matrix_position(point_idx, from_pos, num_neigh)] = distance;
    neigh_idxes[get_matrix_position(point_idx, from_pos, num_neigh)] = neigh_idx;
}

void set_values_for_coordinate(int row, int column, int K,double *neigh_distances_matrix, int *neighs_matrix, double dist){
    int h;
    for (h = 0; h < K; h++){
        double neigh_dist = neigh_distances_matrix[row*K + h];
        if(dist < neigh_dist){
            right_shift_from_position(neighs_matrix,neigh_distances_matrix,K,h,row);
            set_values_to_neigh(neigh_distances_matrix,neighs_matrix,K,dist,row,h,column);
            break;
        }
    }
}




int main(int argc, char *argv[]){

    // ----------------------INIT----------------------

    int N,K;
    int cube_side_value,points_per_process;
    int *neighs_matrix;
    double *neigh_distances_matrix;
    double *distances;
    t_point *points;
    double start, finish;


    // ----------------------CONTROLS----------------------

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


    // ----------------------SETTING UP----------------------

    cube_side_value = 100;

    points = (t_point *) malloc(sizeof(t_point) * N);

    srand(time(0));
    generate_points(points, N, cube_side_value);
    neighs_matrix = (int *)malloc(sizeof(int)*K*N);
    neigh_distances_matrix = (double *)malloc(sizeof(double)*K*N);
    distances = (double *)malloc(sizeof(double)*N*N);
    fill_default_values(neigh_distances_matrix,neighs_matrix,K,N,cube_side_value);


    // ----------------------COMPUTATION----------------------

    start = actual_time();  

    int i,j;
    for (i = 0; i < N; i++){
        for (j = 0; j < N; j++){
            double dist;
            if(i >= j) dist = 0.0;
            else dist = euclideanDistance(&points[i],&points[j]);

            distances[i*N+j] = dist;
        }
    }

    for (i = 0; i < N; i++){
        for (j = 0; j < N; j++){
            if(i >= j) continue;

            double dist = distances[i*N+j];

            //devo fare per i due X punti che sono legati da questo valore
            //esempio questo punto è (Xi,Xj) = 20 -> faccio per Xi e Xj
            set_values_for_coordinate(i,j,K,neigh_distances_matrix,neighs_matrix,dist);
            set_values_for_coordinate(j,i,K,neigh_distances_matrix,neighs_matrix,dist);

        }
    }


    finish = actual_time() - start;


    // ----------------------PRINT----------------------

    
    printf("N=%d , K=%d -> Time elapsed: %lf seconds\n",N,K,finish);
    FILE *results_file = fopen("results.txt","a+");
    fprintf(results_file, "(N=%d) (K=%d)\t	Max Time=%lf\n", N, K, finish);
    fclose(results_file);

    // ----------------------FREE MEMORY----------------------

    
    free(neigh_distances_matrix);
    free(neighs_matrix);
    free(distances);
    free(points);

    return EXIT_SUCCESS;
}
