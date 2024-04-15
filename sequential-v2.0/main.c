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
    int i;
    for (i = 0; i < num_points; i++) {
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

int get_offset(int row , int col, int matrix_dim){
    if(row < col){
        int c = 0;
        int i;
        for (i = 0; i < row; i++) c -=i;
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

double actual_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double sec = tv.tv_sec + tv.tv_usec / 1000000.0;
    return sec;
}

void fill_default_values(double *neigh_distance, int *neigh_idxes,int num_neigh,int num_points){
    int i;
    for (i = 0; i < num_neigh*num_points; i++){
        neigh_distance[i] = 100*sqrt(3) +1;
        neigh_idxes[i] = -1;
    }
}

void set_values_to_neigh(double *neigh_distances, int *neigh_idxes,int num_neigh,double distance,int point_idx,int from_pos,int neigh_idx){
    neigh_distances[get_matrix_position(point_idx, from_pos, num_neigh)] = distance;
    neigh_idxes[get_matrix_position(point_idx, from_pos, num_neigh)] = neigh_idx;
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

int main(int argc, char **argv){
    if(argc < 3 || argc > 4) {
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
    int cube_side_value = 100;
    int *neighs_matrix = (int *)malloc(sizeof(int)*K*N);
    double *neigh_distances_matrix = (double *)malloc(sizeof(double)*K*N);
    srand(time(0));
    (argc == 4) ? load_points_from_file(argv[3], points, N) : generate_points(points, N, cube_side_value);
    fill_default_values(neigh_distances_matrix,neighs_matrix,K,N);

    //TIME
    double tick = actual_time();

    //COMPUTATION
    int i,j,h;
    for (i = 0; i < N; i++){
        for (j = 0; j < N; j++){
            if(i == j) continue;
                double dist = euclideanDistance(&points[i],&points[j]);
                for (h = 0; h < K; h++){
                    double neigh_dist = neigh_distances_matrix[get_matrix_position(i,h,K)];
                    if(dist < neigh_dist){
                        right_shift_from_position(neighs_matrix,neigh_distances_matrix,K,h,i);
                        set_values_to_neigh(neigh_distances_matrix,neighs_matrix,K,dist,i,h,j);
                        break;
                    }
                }
        }
    }
    double finish = actual_time() - tick;
    printf("N=%d , K=%d -> Time elapsed: %lf seconds\n",N,K,finish);

    FILE *results_file = fopen("resultsSequential2-0.txt","a+");
    fprintf(results_file, "(N=%d) (K=%d)\t	Max Time=%lf\n", N, K, finish);
    fclose(results_file);

    free(neigh_distances_matrix);
    free(neighs_matrix);
    free(points);
return EXIT_SUCCESS;
}
