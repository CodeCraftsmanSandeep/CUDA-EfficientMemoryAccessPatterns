// Takes two file paths as command line arguments and calls the compute function from which different access patterns kernels are launched

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

void compute(const unsigned int n, const int* __restrict__ d_C, const int* __restrict__ d_B, long long int* __restrict__ d_dotProduct);

long long int solve(const unsigned int n, const int* __restrict__ C, const int* __restrict__ B)
{
    int* d_C;
    cudaMalloc(&d_C, n * sizeof(int));
    cudaMemcpy(d_C, C, n * sizeof(int), cudaMemcpyHostToDevice);

    int* d_B;
    cudaMalloc(&d_B, n * sizeof(int));
    cudaMemcpy(d_B, B, n * sizeof(int), cudaMemcpyHostToDevice);

    long long int* d_dotProduct;
    cudaMalloc(&d_dotProduct, sizeof(long long int));

    double start_time, end_time, time_consumed;

    for(int run = 1; run <= NUM_RUNS; run++)
    {
        start_time = rtClock();
        compute(n, d_C, d_B, d_dotProduct);
        end_time   = rtClock();

        time_consumed = end_time - start_time;
        execution_times.push_back(time_consumed * 1e3);
    }

    long long int dotProduct;
    cudaMemcpy(&dotProduct, d_dotProduct, sizeof(long long int), cudaMemcpyDeviceToHost);

    cudaFree(d_C);
    cudaFree(d_B);
    cudaFree(d_dotProduct);

    return dotProduct;
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
    if(argc != 3){
        printError(ERROR_FILE); 
        fprintf(ERROR_FILE, "Expected command line argument: input1_file_path input2_file_path\n");
        exit(EXIT_FAILURE);
    }

    const char* input1_file_path    = argv[1];
    const char* input2_file_path    = argv[2];

    // Taking C as input
    // dC ~> dimension of vector C (in this case one-dimensional)
    // nC ~> length of vector

    unsigned int dC, nC;
    int* C;
    {
        FILE* fp = fopen(input1_file_path, "r");
        if(fp == NULL)
        {
            printError(ERROR_FILE);
            fprintf(ERROR_FILE, "Opening %s failed!\n", input1_file_path);
            return 1;
        }

        fscanf(fp, "%u %u", &dC, &nC);

        C = (int*) malloc (nC * sizeof(int));

        for(int i = 0; i < nC; i++){
            fscanf(fp, "%d", &C[i]);
        }

        fclose(fp);
    }

    // Taking matrix B as input
    // dB ~> dimension of B (in this case 1)
    // nB ~> length of vector

    unsigned int dB, nB; 
    int* B;
    {   
        FILE* fp = fopen(input2_file_path, "r");
        if(fp == NULL)
        {   
            printError(ERROR_FILE);
            fprintf(ERROR_FILE, "Opening %s failed!\n", input1_file_path);
            return 1;
        }   

        fscanf(fp, "%u %u", &dB, &nB);

        B = (int*) malloc (nB * sizeof(int));

        for(int i = 0; i < nB; i++){
            fscanf(fp, "%d", &B[i]);
        }

        fclose(fp);
    }

    if(dC != 1 || dB != 1)
    {
        printError(ERROR_FILE);
        fprintf(ERROR_FILE, "Dimension of vectors should be 1\n");
        return 1;
    }

    if(nC != nB)
    {
        printError(ERROR_FILE);
        fprintf(ERROR_FILE, "Length of vectors should be same\n");
        return 1;
    }

    // Solving
    long long int dotProduct = solve(nC, C, B);

    // Printing results
    {
        fprintf(OUTPUT_FILE, "n,dotProduct,mean-time(ms),median-time(ms),std-deviation(ms)\n");
        fprintf(OUTPUT_FILE, "%d,%lld,%.6f,%.6f,%.6f\n", nC, dotProduct, findMean(execution_times), findMedian(execution_times), findStandardDeviation(execution_times));
    }

    free(B);
    free(C);

    return 0;
}
