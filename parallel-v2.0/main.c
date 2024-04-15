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

    MPI_Init(&argc, &argv);

    MPI_Datatype point_type;
    int block_length[] = {1,1,1};
    MPI_Aint displacements[] = {0, sizeof(double), 2*sizeof(double)};
    MPI_Datatype types[3] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
    MPI_Type_create_struct(3,block_length,displacements,types,&point_type);
    MPI_Type_commit(&point_type);

    int my_rank,num_procs,N,K;
    int cube_side_value,points_per_process;
    int *neighs_matrix;
    double *neigh_distances_matrix;
    double *r_buffer_distances;
    int *r_buffer_neighs;
    t_point *points;
    double start, finish;

    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

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

    points_per_process = N / num_procs;
    //SETTING UP
    points = (t_point *) malloc(sizeof(t_point) * N);
    cube_side_value = 100;

    if(my_rank == 0) {
        srand(time(0));
        (argc == 4) ? load_points_from_file(argv[3], points, N) : generate_points(points, N, cube_side_value);
        r_buffer_neighs = (int *)malloc(sizeof(int)*K*N);
        r_buffer_distances = (double *)malloc(sizeof(double)*K*N);
    }
    neighs_matrix = (int *)malloc(sizeof(int)*K*points_per_process);
    neigh_distances_matrix = (double *)malloc(sizeof(double)*K*points_per_process);
    fill_default_values(neigh_distances_matrix,neighs_matrix,K,points_per_process,cube_side_value);

    MPI_Barrier(MPI_COMM_WORLD);
    // TIME
    start = MPI_Wtime();

    MPI_Bcast(points,N,point_type,0,MPI_COMM_WORLD);

    //COMPUTATION
    int i,j,h;
    for (i = points_per_process*my_rank; i < points_per_process*(my_rank+1); i++){
        for (j =  0; j < N; j++){
            if(i == j) continue;

            double dist = euclideanDistance(&points[i],&points[j]);

            for (h = 0; h < K; h++){
                //anche per i neighbours
                double neigh_dist = neigh_distances_matrix[((i % points_per_process) * K) + h];
                if(dist < neigh_dist){
                    right_shift_from_position(neighs_matrix,neigh_distances_matrix,K,h,(i % points_per_process));
                    set_values_to_neigh(neigh_distances_matrix,neighs_matrix,K,dist,(i % points_per_process),h,j);
                    break;
                }
            }
        }
    }
    
    MPI_Gather(neigh_distances_matrix,K*points_per_process,MPI_DOUBLE,r_buffer_distances,K*points_per_process,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Gather(neighs_matrix,K*points_per_process,MPI_INT,r_buffer_neighs,K*points_per_process,MPI_INT,0,MPI_COMM_WORLD);


    finish = MPI_Wtime()-start;
    double max_time;

    MPI_Reduce(&finish,&max_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

    if(my_rank ==0){
        printf("P = %d , N=%d , K=%d -> Time elapsed: %lf seconds\n",num_procs,N,K,max_time);

        // FILE *fp_neighs, *fp_distances;
        // fp_neighs = fopen("neighbours.csv", "w");
        // fp_distances = fopen("min-distances.csv", "w");
        // for (int i=0; i<N; i++) {
        //     for(int j=0; j<K; j++) {
        //         int array_idx = i*K + j;
        //         int neigh_idx = r_buffer_neighs[array_idx];
        //         fprintf(fp_neighs,"[%lf,%lf,%lf]\t", points[neigh_idx].x,points[neigh_idx].y,points[neigh_idx].z);
        //         fprintf(fp_distances,"%lf\t", r_buffer_distances[array_idx]);
        //     }
        //     fprintf(fp_neighs,"\n");
        //     fprintf(fp_distances,"\n");
        // }
        // fclose(fp_neighs);
        // fclose(fp_distances);

        FILE *results_file = fopen("resultsParallel2-0.txt","a+");
        fprintf(results_file, "(P=%d) (N=%d) (K=%d)\t	Max Time=%lf\n",num_procs,N,K,max_time);
        fclose(results_file);
    }
    //Freeing memory
    if(my_rank ==0){
        free(r_buffer_distances);
        free(r_buffer_neighs);
    }
    MPI_Type_free(&point_type);
    free(neigh_distances_matrix);
    free(neighs_matrix);
    free(points);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
