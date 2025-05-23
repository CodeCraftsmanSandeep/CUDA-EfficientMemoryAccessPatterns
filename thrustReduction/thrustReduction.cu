#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>

#define printError(fp) fprintf(fp, "Error in file %s at line: %d\n", __FILE__, __LINE__) 
#define ERROR_FILE stderr
#define OUTPUT_FILE stdout
#define NUM_RUNS 11

void printGPUUsage(const char *msg) {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("%s | Free: %.2f MB, Used: %.2f MB, Total: %.2f MB\n",
           msg,
           free_mem / (1024.0 * 1024.0),
           (total_mem - free_mem) / (1024.0 * 1024.0),
           total_mem / (1024.0 * 1024.0));
}

double rtClock(){
    struct timezone Tzp;
    struct timeval Tp;
    int stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

class result{
private:
    char* file_name;
    size_t N;
    long long int sum;
    double mean_time, median_time, standard_deviation;
    size_t num_runs;

public:
    // Setters 
    void setFileName(char* file_name){
        this->file_name = file_name;
    }
    void setSize(size_t N){
        this->N = N;
    }
    void setSum(long long int sum){
        this->sum = sum;
    }
    void setMeanTime(double mean_time){
        this->mean_time = mean_time;
    }
    void setMedianTime(double median_time){
        this->median_time = median_time;
    }
    void setStandardDeviation(double standard_deviation){
        this->standard_deviation = standard_deviation;
    }
    void setNumRuns(size_t num_runs){
        this->num_runs = num_runs;
    }

    // Getters
    char* getFileName(){
        return file_name;
    }   
    size_t getSize(){
        return N;
    }   
    long long int getSum(){
        return sum;
    }   
    double getMeanTime(){
        return mean_time;
    }   
    double getMedianTime(){
        return median_time;
    }   
    double getStandardDeviation(){
        return standard_deviation;
    }
    size_t getNumRuns(){
        return num_runs;
    }
};

void printResult(result* curr_result){
    fprintf(OUTPUT_FILE, "%s,",     curr_result->getFileName());
    fprintf(OUTPUT_FILE, "%zu,",    curr_result->getSize());
    fprintf(OUTPUT_FILE, "%lld,",     curr_result->getSum());
    fprintf(OUTPUT_FILE, "%zu,",    curr_result->getNumRuns()); 
    fprintf(OUTPUT_FILE, "%.12f,",  curr_result->getMeanTime());
    fprintf(OUTPUT_FILE, "%.12f,",  curr_result->getMedianTime());
    fprintf(OUTPUT_FILE, "%.12f\n", curr_result->getStandardDeviation());
    return;
}

double findMean(const std::vector <double> times){
    if(times.size() == 0) return 0;

    return std::accumulate(times.begin(), times.end(), (double)0) / times.size();
}

double findMedian(std::vector <double> times){
    if(times.size() == 0) return 0;

    std::sort(times.begin(), times.end());
    if(times.size() & 1) return times[times.size()/2];
    return (times[times.size()/2 - 1] + times[times.size()/2])/2;
}

double findStandardDeviation(const std::vector<double> times) {
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

void getResult(const size_t N, int* a, result* curr_result, long long int host_sum){
    // Allocating memory on device
    thrust::device_vector <int> d_a(a, a + N);

    std::vector <double> times;
    long long int sum, curr_sum;

    // Computing Reduction and saving result
    double start_time = rtClock();
    sum = thrust::reduce(d_a.begin(), d_a.end(), 0LL, thrust::plus<long long int> ());
    double end_time = rtClock();
    
    double time_consumed = end_time - start_time;
    times.push_back(time_consumed * 1e3);

    if(sum != host_sum){
        printError(ERROR_FILE);
        fprintf(ERROR_FILE, "Host sum (%lld) != Device sum (%lld)\n", host_sum, sum);
        free(curr_result);
        free(a);
        exit(EXIT_FAILURE);
    }

    for(size_t RUN = 1; RUN < NUM_RUNS; RUN++){
        // Computing Reduction and saving result
        start_time = rtClock();
        curr_sum = thrust::reduce(d_a.begin(), d_a.end(), 0LL, thrust::plus<long long int> ());
        end_time = rtClock();

        time_consumed = end_time - start_time;
        times.push_back(time_consumed * 1e3);
        
        if(curr_sum != sum){
            printError(ERROR_FILE);
            printf("curr_sum = %lld, prev_sum = %lld\n", curr_sum, sum);
            fprintf(ERROR_FILE, "The computation is incorrect!\n");
            free(curr_result);
            exit(EXIT_FAILURE);
        }
    }

    curr_result->setSum(sum);
    curr_result->setMeanTime(findMean(times));
    curr_result->setMedianTime(findMedian(times));
    curr_result->setStandardDeviation(findStandardDeviation(times));

    return;
}

long long int host_reduce(const unsigned int N, int* a){
    long long int host_sum = 0;
    for(unsigned int i = 0; i < N; i++) host_sum += a[i];
    return host_sum;
}

__global__ void wakeUpKernel(){
    // A simple wakeUpKernel
}

int main(const int argc, char* argv[]){
    // Waking up GPU
    wakeUpKernel <<<1, 1>>> ();
    cudaDeviceSynchronize();

    // Checking command line arguments
    if(argc == 1){
        printError(ERROR_FILE); 
        fprintf(ERROR_FILE, "Expected command line argument: input_file_names\n");
        exit(EXIT_FAILURE);
    }

    if(argc > 2){
        printError(ERROR_FILE);
        fprintf(ERROR_FILE, "Practise followed: Only one input is run per execution of this code, because for unknown reasons run of one input is effecting the execution of sub-sequent runs\n");
        exit(EXIT_FAILURE);
    }
    fprintf(OUTPUT_FILE, "File-name,N,Sum,Num-of-runs,Mean-time(ms),Median-time(ms),Standard-deviation(ms)\n"); 

    for(size_t file_num = 1; file_num < argc; file_num++){
        char* input_file_name = argv[file_num];

        // Opening input file for reading input
        FILE* fp = fopen(input_file_name, "r");
        if(fp == NULL){
            printError(ERROR_FILE);
            fprintf(ERROR_FILE, "Opening %s failed!\n", input_file_name);
            continue;
        }

        // Reading array size
        size_t N;
        if(fscanf(fp, "%zu\n", &N) != 1){
            printError(ERROR_FILE);
            fprintf(ERROR_FILE, "Reading array size failed!\n");
            fclose(fp);
            continue;
        }

        // Allocating memory dynamically on host
        int* a = (int*)malloc(N * sizeof(int));
        if(a == NULL){
            printError(ERROR_FILE);
            fprintf(ERROR_FILE, "Memory allocation for %zu elements on host failed!\n", N);
            fclose(fp);
            continue;
        }

        bool reading_failed = false;
        // Reading elements from the file
        for(size_t i = 0; i < N; i++){
            if(fscanf(fp, "%d", &a[i]) != 1){
                printError(ERROR_FILE);
                fprintf(ERROR_FILE, "Reading element %zu from file %s failed!\n", i, input_file_name);
                fclose(fp);
                free(a);
                reading_failed = true;
                break;
            }
        }
        if(reading_failed) continue;

        // Closing the input file
        fclose(fp);

        result* curr_result = (result*)malloc(sizeof(result));
        if(curr_result == NULL){
            printError(ERROR_FILE);
            fprintf(ERROR_FILE, "Memory allocation failed for result!\n");
            free(a);
            continue;
        }
        curr_result->setSize(N);
        curr_result->setFileName(input_file_name);
        curr_result->setNumRuns(NUM_RUNS);

        long long int host_sum = host_reduce(N, a);

        // Calling getResult function
        getResult (N, a, curr_result, host_sum);

        // Calling printResult function
        printResult (curr_result);

        // Deallocating memory on host
        free(a);
        free(curr_result);
    }
    return 0;
}
