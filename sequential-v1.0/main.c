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

void load_points_from_file(char *file_name, t_point *points,int num_points)
{
    FILE *points_file = fopen(file_name, "r");
    if (points_file == NULL)
    {
        perror("Error while opening input file.\n");
        exit(-1);
    }
    int i;
    for (i = 0; i < num_points; i++) fscanf(points_file, "[%lf,%lf,%lf]\n", &points[i].x, &points[i].y, &points[i].z);
    fclose(points_file);
}

int main(int argc, char *argv[]){

    // ----------------------INIT----------------------

    int N,K,N_blocks_per_row;
    int cube_side_value,points_per_process,block_size;
    int *neighs_matrix;
    double *neigh_distances_matrix;
    double *distances;
    t_point *points;
    double start, finish;


    // ----------------------CONTROLS----------------------

    if(argc < 3 || argc > 4) {
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
    block_size = 256;
    N_blocks_per_row = N/block_size;

    points = (t_point *) malloc(sizeof(t_point) * N);

    srand(time(0));
    (argc == 4) ? load_points_from_file(argv[3], points, N) : generate_points(points, N, cube_side_value);
    neighs_matrix = (int *)malloc(sizeof(int)*K*N);
    neigh_distances_matrix = (double *)malloc(sizeof(double)*K*N);
    distances = (double *)malloc(sizeof(double)*block_size*block_size);
    fill_default_values(neigh_distances_matrix,neighs_matrix,K,N,cube_side_value);


    // ----------------------COMPUTATION----------------------

    // printf("--------------NUMBERS--------------");
    // for (int i = 0; i < N; i++){
    //     printf("\n");
    //     printf("X%d:\t%lf\t%lf\t%lf ",i ,points[i].x, points[i].y, points[i].z);
    // }
    // printf("\n");


    start = actual_time();  

    int b_column,b_row,i,j;
    for(b_column = 0; b_column<N_blocks_per_row; b_column++){
        for(b_row = 0; b_row<N_blocks_per_row; b_row++){
        
            for (i = b_column*block_size; i < (b_column+1)*block_size; i++){
                for (j =  b_row*block_size; j <(b_row+1)*block_size; j++){
                    double dist;
                    dist = euclideanDistance(&points[i],&points[j]);
                    //printf("I: %d \t J: %d \t Dist: %lf \n",i,j,dist);
                    //printf("Position -> %d\n",((i%block_size)*block_size)+(j%block_size));

                    distances[((i%block_size)*block_size)+(j%block_size)] = dist;
                }
            }

            // printf("--------------DISTANCES Block (%d,%d)--------------",b_column,b_row);
            // for (int i = 0; i < block_size*block_size; i++){
            //     if(i%block_size == 0) printf("\nX%d: \t",(b_column*block_size)+(i/block_size));
            //     printf("%lf \t ", distances[i]);
            // }
            // printf("\n");

            
            for (i = 0; i < block_size; i++){
                for (j = 0; j < block_size; j++){
                    if(i == j && b_row == b_column) continue;
                    double dist = distances[((i%block_size)*block_size)+(j%block_size)];
                    set_values_for_coordinate(b_column*block_size+i,b_row*block_size+j,K,neigh_distances_matrix,neighs_matrix,dist);
                }
            }

        }
    }


    finish = actual_time() - start;


    // ----------------------PRINT----------------------

    // printf("\n");
    // printf("--------------DISTANCES NEIGH--------------");
    // for (int i = 0; i < N*K; i++){
    //     if(i%K == 0) printf("\nX%d: \t",i/K);
    //     printf("%lf \t ", neigh_distances_matrix[i]);
    // }

    // printf("\n");
    // printf("--------------NEIGH--------------");
    // for (int i = 0; i < N*K; i++){
    //     if(i%K == 0) printf("\nX%d: \t",i/K);
    //     printf("%d \t ", neighs_matrix[i]);
    // }
    // printf("\n");

    printf("N=%d , K=%d -> Time elapsed: %lf seconds\n",N,K,finish);
    FILE *results_file = fopen("resultsSequential1-0.txt","a+");
    fprintf(results_file, "(N=%d) (K=%d)\t	Max Time=%lf\n", N, K, finish);
    fclose(results_file);

    // ----------------------FREE MEMORY----------------------

    
    free(neigh_distances_matrix);
    free(neighs_matrix);
    free(distances);
    free(points);

    return EXIT_SUCCESS;
}
