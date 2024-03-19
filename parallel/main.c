#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdbool.h>
#include "mpi.h"

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

void print_error_argc(int argc){
    fprintf(stderr, "\n\nWrong number of parameters! Expected 3, but are %d!\n", argc);
    fprintf(stderr, "\n\nValid parameters are knnmpi {number_of_points} {number_of_neighbours}\n");
}

void print_error_neighbours(int points_number, int neighbours_number){
    fprintf(stderr, "\n\nNeighbours to find are more than the total number of points! Expected < %d, but are %d!\n",points_number,neighbours_number);
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
                        for (int r = K - 1; r > h; r--){
                            neigh_distances_matrix[i*K +r] = neigh_distances_matrix[i*K +(r - 1)];
                            neighs_matrix[i*K +r] = neighs_matrix[i*K +(r - 1)];
                        }
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
return EXIT_SUCCESS;
}
