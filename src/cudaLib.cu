
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size){
		y[idx] = scale * x[idx] + y[idx];
	}
}

int runGpuSaxpy(int vectorSize) {
	std::cout << "Hello GPU Saxpy!\n";
	std::cout << "Adding vectors \n";

	std::srand(static_cast<unsigned int>(std::time(0)));

	//	Insert code here
	float *h_x = new float[vectorSize];
	float *h_y = new float[vectorSize];
	float *h_y_cpu = new float[vectorSize];
	float scale = 2.0f;
	float *d_x, *d_y;

	for (int i = 0; i < vectorSize; i++) {
        h_x[i] = static_cast<float>(std::rand()) / RAND_MAX * 10.0f;  // Random float between 0 and 10
        h_y[i] = static_cast<float>(std::rand()) / RAND_MAX * 10.0f;  // Random float between 0 and 10
    }
	scale = static_cast<float>(std::rand()) / RAND_MAX * 10.0f;
	std::cout << "Scale is " << scale << "\n\n";
	std::cout << "First 5 of vector X: ";
	for(int i = 0; i < 5 && i < vectorSize; i++){
		std::cout << h_x[i] << " ";
	}
	std::cout << "\n\n";
	std::cout << "First 5 of vector Y: ";
	for(int i = 0; i < 5 && i < vectorSize; i++){
		std::cout << h_y[i] << " ";
		
	}

	
	cudaMalloc((void**)&d_x, vectorSize * sizeof(float));
	cudaMalloc((void**)&d_y, vectorSize * sizeof(float));

	cudaMemcpy(d_x, h_x, vectorSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, vectorSize * sizeof(float), cudaMemcpyHostToDevice);

	int threadPerBlock = 256;
	int blocksPerGrid = (vectorSize + threadPerBlock - 1) / threadPerBlock;

	saxpy_gpu<<<blocksPerGrid, threadPerBlock>>>(d_x, d_y, scale, vectorSize);

	gpuAssert(cudaGetLastError(), __FILE__, __LINE__, true);

	cudaMemcpy(h_y, d_y, vectorSize * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < vectorSize; i++) {
		h_y_cpu[i] = scale * h_x[i] + h_y[i];
	}
	
	std::cout <<"\n\n";
	std::cout << "First 5 values of output vector Y: ";
        for (int i = 0; i < 5 && i < vectorSize; i++) {
		std:: cout << h_y[i] << " ";
    	}
	std:: cout << "\n\n";
	int errorCount = verifyVector(h_x, h_y, h_y_cpu, scale, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

	cudaFree(d_x);
	cudaFree(d_y);
	delete[] h_x;
	delete[] h_y;
	delete[] h_y_cpu;

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
/bin/bash: line 1: wq: command not found

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here
	int idx = blockIdx.x * blockDim.x + threadIdx.x;	
	if (idx < pSumSize){
	uint64_t hitCount = 0;
	curandState rngState;
	curand_init(clock64(), idx, 0, &rngState);
	for (int i = 0; i < sampleSize; i++){
		float x = curand_uniform(&rngState);
		float y = curand_uniform(&rngState);
		if (x * x + y * y <= 1.0f){
			hitCount++;
		}
	}
	pSums[idx] = hitCount;
	}
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < reduceSize){
		uint64_t total = 0;
		for (int i = 0; i < pSumSize; i++){
			total += pSums[idx * pSumSize + i];
		}
		totals[idx] = total;
	}
}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	//      Insert code here
	uint64_t *d_pSums, *d_totals;
	int threadPerBlock = 256;
	int blocksPerGrid = (generateThreadCount + threadPerBlock - 1) / threadPerBlock;
	int reduceThreadPerBlock = 256;
	int reduceBlocksPerGrid = (reduceThreadCount + threadPerBlock - 1) / threadPerBlock;
	
	//cudaMalloc((void**)&d_x, vectorSize * sizeof(float));
	cudaMalloc((void**)&d_pSums, generateThreadCount * sampleSize * sizeof(uint64_t));
	cudaMalloc((void**)&d_totals, reduceThreadCount * sizeof(uint64_t));

	//Run Kernel for Generating Points
	generatePoints<<<blocksPerGrid, threadPerBlock>>>(d_pSums, generateThreadCount, sampleSize);
	gpuAssert(cudaGetLastError(), __FILE__, __LINE__, true);

	//Run Kernel for Reducing Count
	reduceCounts<<<reduceBlocksPerGrid, reduceThreadPerBlock>>>(d_pSums, d_totals, generateThreadCount, reduceSize);
	gpuAssert(cudaGetLastError(), __FILE__, __LINE__, true);

	//Copy GPU to CPU
	uint64_t *h_totals = new uint64_t[reduceThreadCount];
	cudaMemcpy(h_totals, d_totals, reduceThreadCount * sizeof(uint64_t), cudaMemcpyDeviceToHost);

	uint64_t totalHits = 0;
	for(uint64_t i = 0; i < reduceSize; i++){
		totalHits += h_totals[i];
	}

	uint64_t totalSamples = generateThreadCount * sampleSize;
	approxPi = 4.0 * totalHits / totalSamples;

	cudaFree(d_pSums);
	cudaFree(d_totals);

	return approxPi;
}
