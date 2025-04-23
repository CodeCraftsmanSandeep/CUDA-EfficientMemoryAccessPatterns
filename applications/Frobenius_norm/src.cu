// Takes one file path as command line argument and calls the compute function from which different access patterns kernels are launched

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>
#include <algorithm>
#include <numeric>

#define printError(fp) fprintf(fp, "Error in file %s at line: %d\n", __FILE__, __LINE__) 
#define ERROR_FILE stderr
#define OUTPUT_FILE stdout

void printGPUUsage(const char *msg) 
{
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("%s | Free: %.2f MB, Used: %.2f MB, Total: %.2f MB\n",
           msg,
           free_mem / (1024.0 * 1024.0),
           (total_mem - free_mem) / (1024.0 * 1024.0),
           total_mem / (1024.0 * 1024.0));
}

double rtClock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

double findMean(const std::vector <double> times)
{
    if(times.size() == 0) return 0;

    return std::accumulate(times.begin(), times.end(), (double)0) / times.size();
}

double findMedian(std::vector <double> times)
{
    if(times.size() == 0) return 0;

    std::sort(times.begin(), times.end());
    if(times.size() & 1) return times[times.size()/2];
    return (times[times.size()/2 - 1] + times[times.size()/2])/2;
}

double findStandardDeviation(const std::vector<double> times) 
{
    if(times.size() <= 1) return 0;

    // Calculate mean
    double mean = findMean(times);

    // Calculate variance
    double variance = 0.0;
    for (double x : times) {
        variance += (x - mean) * (x - mean);
    }
    variance /= (times.size() - 1);

    // Return standard deviation
    return std::sqrt(variance);
}

const unsigned int NUM_RUNS = 11;
std::vector <double> execution_times;

void compute(const unsigned int m, const unsigned int n, const int* __restrict__ d_C, long long int* __restrict__ d_temp_storage, float* __restrict__ d_Frobenius_norm);

float solve(const unsigned int m, const unsigned int n, const int* __restrict__ C)
{
    int* d_C;
    cudaMalloc(&d_C, m * n * sizeof(int));
    cudaMemcpy(d_C, C, m * n * sizeof(int), cudaMemcpyHostToDevice);

    long long int* d_temp_storage;
    cudaMalloc(&d_temp_storage, sizeof(long long int));

    float* d_Frobenius_norm;
    cudaMalloc(&d_Frobenius_norm, sizeof(float));

    double start_time, end_time, time_consumed;

    for(int run = 1; run <= NUM_RUNS; run++)
    {
        start_time = rtClock();
        compute(m, n, d_C, d_temp_storage, d_Frobenius_norm);
        end_time   = rtClock();

        time_consumed = end_time - start_time;
        execution_times.push_back(time_consumed * 1e3);
    }

    float Frobenius_norm;
    cudaMemcpy(&Frobenius_norm, d_Frobenius_norm, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_temp_storage);
    cudaFree(d_C);
    cudaFree(d_Frobenius_norm);

    return Frobenius_norm;
}

__device__ int device_var;
__global__ void wakeUpKernel(){
    // A simple wakeUpKernel
    device_var = 2 * device_var * 100;
}

int main(const int argc, char* argv[]){
    // Waking up GPU
    wakeUpKernel <<<1, 1>>> ();
    cudaDeviceSynchronize();

    // Checking command line arguments
    if(argc != 2){
        printError(ERROR_FILE); 
        fprintf(ERROR_FILE, "Expected command line argument: input_file_path\n");
        exit(EXIT_FAILURE);
    }

    const char* input_file_path    = argv[1];

    // Taking C as input
    // dC ~> dimension of vector C (in this case two-dimensional)
    // (mC, nC) ~> size of 2-d vector

    unsigned int dC, mC, nC;
    int* C;
    {
        FILE* fp = fopen(input_file_path, "r");
        if(fp == NULL)
        {
            printError(ERROR_FILE);
            fprintf(ERROR_FILE, "Opening %s failed!\n", input_file_path);
            return 1;
        }

        fscanf(fp, "%u %u %u", &dC, &mC, &nC);

        C = (int*) malloc (mC * nC * sizeof(int));

        for(int i = 0; i < mC; i++){
            for(int j = 0; j < nC; j++){
                fscanf(fp, "%d", &C[i * nC + j]);
            }
        }

        fclose(fp);
    }

    if(dC != 2)
    {
        printError(ERROR_FILE);
        fprintf(ERROR_FILE, "Dimension of vector should be 2\n");
        return 1;
    }

    // Solving
    float Frobenius_norm = solve(mC, nC, C);

    // Printing results
    {
        fprintf(OUTPUT_FILE, "numRows,numCols,matrixSize,Frobenius_norm,mean-time(ms),median-time(ms),std-deviation(ms)\n");
        fprintf(OUTPUT_FILE, "%d,%d,%d,%.6f,%.6f,%.6f,%.6f\n", mC, nC, mC * nC, Frobenius_norm, findMean(execution_times), findMedian(execution_times), findStandardDeviation(execution_times));
    }

    free(C);

    return 0;
}
