#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <stdbool.h>
#include <mpi.h>

#define RECEIVED_TAG 1
#define MYPOINT_TAG 0
#define COORDINATOR 0

typedef struct point_st
{
    double x;
    double y;
    double z;
} t_point;

void print_error_argc(int argc)
{
    fprintf(stderr, "\n\nWrong number of parameters! Expected 3, but are %d!\n", argc);
    fprintf(stderr, "\n\nValid parameters are knnmpi {number_of_points} {number_of_neighbours}\n");
}

void print_error_neighbours(int points_number, int neighbours_number)
{
    fprintf(stderr, "\n\nNeighbours to find are more than the total number of points! Expected < %d, but are %d!\n", points_number, neighbours_number);
}

double euclideanDistance(t_point *point1, t_point *point2)
{
    double x = point1->x - point2->x;
    double y = point1->y - point2->y;
    double z = point1->z - point2->z;
    return sqrt(x * x + y * y + z * z);
}

void generate_points(t_point *points, int num_points, int cube_length)
{
    int i;
    for (i = 0; i < num_points; i++)
    {
        points[i].x = (double)rand() / RAND_MAX * cube_length;
        points[i].y = (double)rand() / RAND_MAX * cube_length;
        points[i].z = (double)rand() / RAND_MAX * cube_length;
    }
}

void fill_default_values(double *neigh_distance, int *neigh_idxes, int num_neigh, int num_points, int cube_dim)
{
    int i;
    for (i = 0; i < num_neigh * num_points; i++)
    {
        neigh_distance[i] = cube_dim * sqrt(3) + 1;
        neigh_idxes[i] = -1;
    }
}

int find_position(double *array, int left, int right, double to_insert)
{
    while (left <= right)
    {
        int mid = (left + right) / 2;

        if (array[mid] <= to_insert)
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

void insert_value(double *neigh_dists, int *neigh_idxs, int neighs_number, double distance_to_insert, int neigh_to_insert, int idx_point)
{
    int position = find_position(neigh_dists, idx_point * neighs_number, (idx_point + 1) * neighs_number - 1, distance_to_insert);
    printf("poition %d for distance\n",position);
    if(position <= (idx_point + 1) * neighs_number - 1){
        int i;
        for (i = (idx_point + 1) * neighs_number - 1; i > position; i--)
        {
            neigh_dists[i] = neigh_dists[i - 1];
            neigh_idxs[i] = neigh_idxs[i - 1];
        }
        neigh_dists[position] = distance_to_insert;
        neigh_idxs[position] = neigh_to_insert;
    }
}

void load_points_from_file(char *file_name, t_point *points, int num_points)
{
    FILE *points_file = fopen(file_name, "r");
    if (points_file == NULL)
    {
        perror("Error while opening input file.\n");
        exit(-1);
    }
    int i;
    for (i = 0; i < num_points; i++)
        fscanf(points_file, "[%lf,%lf,%lf]\n", &points[i].x, &points[i].y, &points[i].z);
    fclose(points_file);
}

void calculate_and_insert_distance(double *where_insert_distance,int *where_insert_neigh,int get_from,int number_of_points,int number_of_neigh,t_point *my_points,t_point *received_points,bool skip_diagonal){
    for (int i = 0; i < number_of_points; i++)
    {
        for (int j = 0; j < number_of_points; j++)
        {
            if(i==j && skip_diagonal== true) continue;
            double dist = euclideanDistance(&my_points[i], &received_points[j]);
            insert_value(where_insert_distance, where_insert_neigh, number_of_neigh, dist, j+(get_from*number_of_points), i);
        }
    }
}

int main(int argc, char *argv[])
{

    MPI_Init(&argc, &argv);

    MPI_Datatype point_type;
    int block_length[] = {1, 1, 1};
    MPI_Aint displacements[] = {0, sizeof(double), 2 * sizeof(double)};
    MPI_Datatype types[3] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    MPI_Type_create_struct(3, block_length, displacements, types, &point_type);
    MPI_Type_commit(&point_type);

    int my_rank, num_procs, N, K;
    int cube_side_value, points_per_process;
    int *neighs_matrix;
    double *neigh_distances_matrix;
    double *r_buffer_distances;
    int *r_buffer_neighs;
    t_point *points, *my_points, *received_points;
    int source, dest;
    double start, finish;

    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (argc < 3 || argc > 4)
    {
        print_error_argc(argc);
        return -1;
    }
    N = atoi(argv[1]);
    K = atoi(argv[2]);
    if (K >= N)
    {
        print_error_neighbours(N, K);
        return -1;
    }
    points_per_process = N / num_procs;
    // SETTING UP

    cube_side_value = 100;

    if (my_rank == COORDINATOR)
    {
        points = (t_point *)malloc(sizeof(t_point) * N);
        srand(time(0));
        (argc == 4) ? load_points_from_file(argv[3], points, N) : generate_points(points, N, cube_side_value);
        r_buffer_neighs = (int *)malloc(sizeof(int) * K * N);
        r_buffer_distances = (double *)malloc(sizeof(double) * K * N);
    }
    neighs_matrix = (int *)malloc(sizeof(int) * K * points_per_process);
    neigh_distances_matrix = (double *)malloc(sizeof(double) * K * points_per_process);
    fill_default_values(neigh_distances_matrix, neighs_matrix, K, points_per_process, cube_side_value);
    received_points = (t_point *)malloc(sizeof(t_point) * points_per_process);
    my_points = (t_point *)malloc(sizeof(t_point) * points_per_process);

    dest = (my_rank + 1) % (num_procs);               /* next in a ring */
    source = (my_rank - 1 + num_procs) % (num_procs); /* precedent in a ring */

    MPI_Barrier(MPI_COMM_WORLD);
    // TIME
    start = MPI_Wtime();

    if (my_rank == COORDINATOR)
    {
        MPI_Scatter(points, points_per_process, point_type, my_points, points_per_process, point_type, 0, MPI_COMM_WORLD);
        MPI_Scatter(points, points_per_process, point_type, received_points, points_per_process, point_type, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Scatter(NULL, points_per_process, point_type, my_points, points_per_process, point_type, 0, MPI_COMM_WORLD);
        MPI_Scatter(NULL, points_per_process, point_type, received_points, points_per_process, point_type, 0, MPI_COMM_WORLD);
    }

    calculate_and_insert_distance(neigh_distances_matrix,neighs_matrix,points_per_process,K,my_points,received_points,true);
    // // MPI_Send(received_points, points_per_process, point_type, dest, RECEIVED_TAG, MPI_COMM_WORLD);
    // // MPI_Recv(received_points, points_per_process, point_type, source, RECEIVED_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    int p = 1;
    while(p < num_procs)
    {
        int buffer_attached_size = MPI_BSEND_OVERHEAD + sizeof(t_point)*points_per_process;
        char *buffer_attached = (char *)malloc(buffer_attached_size);
        MPI_Buffer_attach(buffer_attached, buffer_attached_size);
        //MPI_Request req;
        MPI_Bsend(received_points, points_per_process, point_type, dest, RECEIVED_TAG, MPI_COMM_WORLD);
        MPI_Recv(received_points, points_per_process, point_type, source, RECEIVED_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        MPI_Buffer_detach(&buffer_attached, &buffer_attached_size);
        free(buffer_attached);

        calculate_and_insert_distance(neigh_distances_matrix,neighs_matrix,(my_rank - p + num_procs) % (num_procs),points_per_process,K,my_points,received_points,false);

        p++;
    }
    
   

    MPI_Gather(neigh_distances_matrix, K * points_per_process, MPI_DOUBLE, r_buffer_distances, K * points_per_process, MPI_DOUBLE, COORDINATOR, MPI_COMM_WORLD);
    MPI_Gather(neighs_matrix, K * points_per_process, MPI_INT, r_buffer_neighs, K * points_per_process, MPI_INT, COORDINATOR, MPI_COMM_WORLD);

    // COMPUTATION
    //  int i,j;
    //  for (i = points_per_process*my_rank; i < points_per_process*(my_rank+1); i++){
    //      for (j = 0; j < N; j++){
    //          if(i == j) continue;

    //         double dist = euclideanDistance(&points[i],&points[j]);

    //         insert_value(neigh_distances_matrix,neighs_matrix,K,dist,j,(i % points_per_process));

    //     }
    // }

    // MPI_Gather(neigh_distances_matrix,K*points_per_process,MPI_DOUBLE,r_buffer_distances,K*points_per_process,MPI_DOUBLE,0,MPI_COMM_WORLD);
    // MPI_Gather(neighs_matrix,K*points_per_process,MPI_INT,r_buffer_neighs,K*points_per_process,MPI_INT,0,MPI_COMM_WORLD);

    finish = MPI_Wtime() - start;
    double max_time;

    MPI_Reduce(&finish, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);


    // PRINTING EXECUTION TIME BY COORDINATOR
    if (my_rank == COORDINATOR)
    {

        // for (int i = 0; i < points_per_process; i++)
        // {
        //     for (int j = 0; j < K; j++)
        //     {
        //         int array_idx = i * K + j;
        //         printf("%lf\t", neigh_distances_matrix[array_idx]);
        //     }
        //     printf("\n");
        // }

        printf("P = %d , N=%d , K=%d -> Time elapsed: %lf seconds\n", num_procs, N, K, max_time);
        // WRITING POINTS , DISTANCES AND NEIGHBOURS INDEX IN DIFFERENT FILES BY COORDINATOR
        // FILE *fp_neighs, *fp_distances;
        // fp_neighs = fopen("neighbours.csv", "w");
        // fp_distances = fopen("min-distances.csv", "w");
        // for (int i=0; i<N; i++) {
        //     for(int j=0; j<K; j++) {
        //         int array_idx = i*K + j;
        //         int neigh_idx = r_buffer_neighs[array_idx];
        //         fprintf(fp_neighs,"%d\t", r_buffer_neighs[array_idx]);
        //         fprintf(fp_distances,"%lf\t", r_buffer_distances[array_idx]);
        //     }
        //     fprintf(fp_neighs,"\n");
        //     fprintf(fp_distances,"\n");
        // }
        // fclose(fp_neighs);
        // fclose(fp_distances);

        FILE *results_file = fopen("resultsParallel2-2-ring.txt","a+");
        fprintf(results_file, "(P=%d) (N=%d) (K=%d)\t	Max Time=%lf\n",num_procs,N,K,max_time);
        fclose(results_file);
    }
    // Freeing memory
    if (my_rank == 0)
    {
        free(r_buffer_distances);
        free(r_buffer_neighs);
        free(points);
    }
    MPI_Type_free(&point_type);
    free(neigh_distances_matrix);
    free(neighs_matrix);
    free(my_points);
    free(received_points);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
