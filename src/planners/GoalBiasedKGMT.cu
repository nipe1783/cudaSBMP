#include "planners/GoalBiasedKGMT.cuh"



#define SAMPLE_DIM 7
#define STATE_DIM 4
#define NUM_R2 8
#define NUM_R1 16
#define NUM_R1_CELLS 256
#define NUM_R2_CELLS 16*16*8*8


GoalBiasedKGMT::GoalBiasedKGMT(float width, float height, int N, int n, int numIterations, int maxTreeSize, int numDisc, float agentLength, float goalThreshold):
    width_(width), height_(height), N_(N), n_(n), numIterations_(numIterations), maxTreeSize_(maxTreeSize), numDisc_(numDisc), agentLength_(agentLength), goalThreshold_(goalThreshold){

    R1Size_ = width / N;
    R2Size_ = width / (n * N);

    OccupancyGrid grid(width, height, N);
    std::vector<int> fromNodes = grid.constructFromNodes();
    std::vector<int> toNodes = grid.constructToNodes();
    nR1Edges_ = fromNodes.size();
    printf("nR1Edges: %d\n", nR1Edges_);
    for(int i = 0; i < nR1Edges_; i++){
        printf("i: %d, from: %d, to: %d\n", i+1, fromNodes[i], toNodes[i]);
    }

    // Allocate memory for the hash map
    tableSize_ = 2 * nR1Edges_; // double the size to reduce collisions
    thrust::device_vector<int> hashTable(2 * tableSize_, -1);
    thrust::device_vector<int> edgeIndices(nR1Edges_);
    thrust::sequence(edgeIndices.begin(), edgeIndices.end());

    // Copy data to device
    d_fromNodes_ = fromNodes;
    d_toNodes_ = toNodes;
    d_edgeIndices_ = edgeIndices;
    d_hashTable_ = hashTable;

    // Initialize the hash map
    initHashMap<<<(nR1Edges_ + 255) / 256, 256>>>(
        thrust::raw_pointer_cast(d_fromNodes_.data()),
        thrust::raw_pointer_cast(d_toNodes_.data()),
        thrust::raw_pointer_cast(d_edgeIndices_.data()),
        thrust::raw_pointer_cast(d_hashTable_.data()),
        tableSize_,
        nR1Edges_);

    cudaDeviceSynchronize();

    d_fromNodes_ptr_ = thrust::raw_pointer_cast(d_fromNodes_.data());
    d_toNodes_ptr_ = thrust::raw_pointer_cast(d_toNodes_.data());
    d_edgeIndices_ptr_ = thrust::raw_pointer_cast(d_edgeIndices_.data());
    d_hashTable_ptr_ = thrust::raw_pointer_cast(d_hashTable_.data());

    d_G_ = thrust::device_vector<bool>(maxTreeSize);
    d_GNew_ = thrust::device_vector<bool>(maxTreeSize);
    d_scanIdx_ = thrust::device_vector<int>(maxTreeSize);
    d_R1scanIdx_ = thrust::device_vector<int>(NUM_R1_CELLS);
    d_activeIdx_ = thrust::device_vector<int>(maxTreeSize);
    d_treeParentIdx_ = thrust::device_vector<int>(maxTreeSize);
    d_treeSamples_ = thrust::device_vector<float>(maxTreeSize * SAMPLE_DIM);
    d_xGoal_ = thrust::device_vector<float>(SAMPLE_DIM);
    d_unexploredSamples_ = thrust::device_vector<float>(maxTreeSize * SAMPLE_DIM);
    d_uParentIdx_ = thrust::device_vector<int>(maxTreeSize);
    d_R1Avail_ = thrust::device_vector<int>(N*N);
    d_R2Avail_ = thrust::device_vector<int>(N*N*n*n);
    d_R1Score_ = thrust::device_vector<float>(N*N);
    d_R1Valid_ = thrust::device_vector<int>(N*N);
    d_R2Valid_ = thrust::device_vector<int>(N*N*n*n);
    d_R1_ = thrust::device_vector<int>(N*N);
    d_R2_ = thrust::device_vector<int>(N*N*n*n);
    d_costs_ = thrust::device_vector<float>(maxTreeSize);
    d_R1EdgeCosts_ = thrust::device_vector<float>(nR1Edges_);
    d_selR1Edge_ = thrust::device_vector<int>(nR1Edges_);
    d_connR1Edge_ = thrust::device_vector<float>(nR1Edges_);
    d_fromR1_ = thrust::device_vector<int>(nR1Edges_);
    d_toR1_ = thrust::device_vector<int>(nR1Edges_);
    d_valR1Edge_ = thrust::device_vector<int>(nR1Edges_);

    d_G_ptr_ = thrust::raw_pointer_cast(d_G_.data());
    d_GNew_ptr_ = thrust::raw_pointer_cast(d_GNew_.data());
    d_treeSamples_ptr_ = thrust::raw_pointer_cast(d_treeSamples_.data());
    d_scanIdx_ptr_ = thrust::raw_pointer_cast(d_scanIdx_.data());
    d_R1scanIdx_ptr_ = thrust::raw_pointer_cast(d_R1scanIdx_.data());
    d_activeIdx_ptr_ = thrust::raw_pointer_cast(d_activeIdx_.data());
    d_treeParentIdx_ptr_ = thrust::raw_pointer_cast(d_treeParentIdx_.data());
    d_xGoal_ptr_ = thrust::raw_pointer_cast(d_xGoal_.data());
    d_unexploredSamples_ptr_ = thrust::raw_pointer_cast(d_unexploredSamples_.data());
    d_uParentIdx_ptr_ = thrust::raw_pointer_cast(d_uParentIdx_.data());
    d_R1Score_ptr_ = thrust::raw_pointer_cast(d_R1Score_.data());
    d_R1Avail_ptr_ = thrust::raw_pointer_cast(d_R1Avail_.data());
    d_R2Avail_ptr_ = thrust::raw_pointer_cast(d_R2Avail_.data());
    d_R1_ptr_ = thrust::raw_pointer_cast(d_R1_.data());
    d_R2_ptr_ = thrust::raw_pointer_cast(d_R2_.data());
    d_R1Valid_ptr_ = thrust::raw_pointer_cast(d_R1Valid_.data());
    d_R2Valid_ptr_ = thrust::raw_pointer_cast(d_R2Valid_.data());
    d_costs_ptr_ = thrust::raw_pointer_cast(d_costs_.data());
    d_R1EdgeCosts_ptr_ = thrust::raw_pointer_cast(d_R1EdgeCosts_.data());
    d_selR1Edge_ptr_ = thrust::raw_pointer_cast(d_selR1Edge_.data());
    d_connR1Edge_ptr_ = thrust::raw_pointer_cast(d_connR1Edge_.data());
    d_fromR1_ptr_ = thrust::raw_pointer_cast(d_fromR1_.data());
    d_toR1_ptr_ = thrust::raw_pointer_cast(d_toR1_.data());
    d_valR1Edge_ptr_ = thrust::raw_pointer_cast(d_valR1Edge_.data());


    cudaMalloc(&d_costToGoal, sizeof(float));
    thrust::fill(d_treeParentIdx_.begin(), d_treeParentIdx_.end(), -1);
    thrust::fill(d_uParentIdx_.begin(), d_uParentIdx_.end(), -1);
    thrust::fill(d_R1Score_.begin(), d_R1Score_.end(), 1.0);

    R1Threshold_ = 0.0;
    cudaMalloc(&d_R1Threshold_ptr_, sizeof(float));
    cudaMemcpy(d_R1Threshold_ptr_, &R1Threshold_, sizeof(float), cudaMemcpyHostToDevice);

    thrust::copy(fromNodes.begin(), fromNodes.end(), d_fromR1_.begin());
    thrust::copy(toNodes.begin(), toNodes.end(), d_toR1_.begin());
}

void GoalBiasedKGMT::plan(float* initial, float* goal, float* d_obstacles, int obstaclesCount) {
    
    double t_kgmtStart = std::clock();
    
    // initialize vectors with root of tree
    cudaMemcpy(d_treeSamples_ptr_, initial, SAMPLE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    bool value = true;
    cudaMemcpy(d_G_ptr_, &value, sizeof(bool), cudaMemcpyHostToDevice);
    int r1_0 = getR1_gb(initial[0], initial[1], R1Size_, N_);
    int r2_0 = getR2_gb(initial[0], initial[1], r1_0, R1Size_, N_, R2Size_, n_);
    thrust::device_ptr<int> d_R1_ptr = d_R1_.data();
    thrust::device_ptr<int> d_R1Avail_ptr = d_R1Avail_.data();
    thrust::device_ptr<int> d_R2Avail_ptr = d_R2Avail_.data();
    thrust::device_ptr<int> d_R1Valid_ptr = d_R1Valid_.data();
    thrust::fill(d_R1_ptr + r1_0, d_R1_ptr + r1_0 + 1, 1);
    thrust::fill(d_R1Avail_ptr + r1_0, d_R1Avail_ptr + r1_0 + 1, 1);
    thrust::fill(d_R2Avail_ptr + r2_0, d_R2Avail_ptr + r2_0 + 1, 1);
    thrust::fill(d_R1Valid_ptr + r1_0, d_R1Valid_ptr + r1_0 + 1, 1);

    // initialize xGoal
    printf("Goal: %f, %f\n", goal[0], goal[1]);
    cudaMemcpy(d_xGoal_ptr_, goal, SAMPLE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    
    const int blockSize = 128;
	const int gridSize = std::min((maxTreeSize_ + blockSize - 1) / blockSize, 2147483647);
    int gridSizeActive = 1;
    int blockSizeActive = 32;

    // initialize random seed for curand
    curandState* d_randomStates;
    cudaMalloc(&d_randomStates, maxTreeSize_ * sizeof(curandState));
    initCurandStates_gb<<<(maxTreeSize_ + blockSize - 1) / blockSize, blockSize>>>(d_randomStates, maxTreeSize_, time(NULL));

    int itr = 0;
    treeSize_ = 1;
    int activeSize = 0;
    int maxIndex;
    float maxValue;
    while(itr < numIterations_){
        itr++;

        // UPDATE GRID SCORES:
        thrust::exclusive_scan(d_R1Avail_.begin(), d_R1Avail_.end(), d_R1scanIdx_.begin(), 0, thrust::plus<int>());
        activeSize = d_R1scanIdx_[N_*N_-1];
        gridSizeActive = int(ceil(float(nR1Edges_) / float(NUM_R1_CELLS)));
        updateR_gb<<<gridSizeActive, NUM_R1_CELLS>>>(
            d_R1Score_ptr_,
            d_R2Avail_ptr_,
            d_R1Avail_ptr_,
            d_R1Valid_ptr_,
            d_R1_ptr_,
            n_, 
            0.01, 
            d_R1EdgeCosts_ptr_,
            activeSize,
            d_fromR1_ptr_,
            d_toR1_ptr_,
            nR1Edges_,
            d_selR1Edge_ptr_,
            d_valR1Edge_ptr_,
            d_R1Threshold_ptr_);

        // PROPAGATE G:
        thrust::exclusive_scan(d_G_.begin(), d_G_.end(), d_scanIdx_.begin(), 0, thrust::plus<int>());
        activeSize = d_scanIdx_[maxTreeSize_-1];
        (d_G_[maxTreeSize_ - 1]) ? ++activeSize : 0;
        
        findInd_gb<<<gridSize, blockSize>>>(
            maxTreeSize_, 
            d_G_ptr_, 
            d_scanIdx_ptr_, 
            d_activeIdx_ptr_);

        

        blockSizeActive = 32;
        gridSizeActive = activeSize;
        if(blockSizeActive*gridSizeActive > maxTreeSize_ - treeSize_){
            blockSizeActive = 128;
            gridSizeActive = int(floor(maxTreeSize_ / blockSizeActive));
            int remaining = maxTreeSize_ - treeSize_;
            int iterations = int(float(remaining) / float(activeSize));
            propagateGV2_gb<<<gridSizeActive, blockSizeActive>>>(
                activeSize, 
                d_activeIdx_ptr_, 
                d_G_ptr_,
                d_GNew_ptr_,
                d_treeSamples_ptr_, 
                d_unexploredSamples_ptr_,
                d_uParentIdx_ptr_,
                d_R1Valid_ptr_,
                d_R2Valid_ptr_,
                d_R1_ptr_,
                d_R2_ptr_,
                d_R1Avail_ptr_,
                d_R2Avail_ptr_,
                N_,
                n_,
                R1Size_,
                R2Size_,
                d_randomStates, 
                numDisc_, 
                agentLength_,
                d_R1Threshold_ptr_,
                d_R1Score_ptr_,
                d_obstacles,
                obstaclesCount,
                iterations,
                width_,
                height_,
                d_selR1Edge_ptr_,
                d_valR1Edge_ptr_);
        }
        else{
            propagateG_gb<<<gridSizeActive, blockSizeActive>>>(
                activeSize, 
                d_activeIdx_ptr_, 
                d_G_ptr_,
                d_GNew_ptr_,
                d_treeSamples_ptr_, 
                d_unexploredSamples_ptr_,
                d_uParentIdx_ptr_,
                d_R1Valid_ptr_,
                d_R2Valid_ptr_,
                d_R1_ptr_,
                d_R2_ptr_,
                d_R1Avail_ptr_,
                d_R2Avail_ptr_,
                N_,
                n_,
                R1Size_,
                R2Size_,
                d_randomStates, 
                numDisc_, 
                agentLength_,
                d_R1Threshold_ptr_,
                d_R1Score_ptr_,
                d_obstacles,
                obstaclesCount,
                width_,
                height_,
                d_selR1Edge_ptr_,
                d_valR1Edge_ptr_,
                d_hashTable_ptr_,
                tableSize_);
        }

        // UPDATE G:
        thrust::exclusive_scan(d_GNew_.begin(), d_GNew_.end(), d_scanIdx_.begin(), 0, thrust::plus<int>());
        activeSize = d_scanIdx_[maxTreeSize_-1];
        (d_GNew_[maxTreeSize_ - 1]) ? ++activeSize : 0;
        findInd_gb<<<gridSize, blockSize>>>(
            maxTreeSize_, 
            d_GNew_ptr_, 
            d_scanIdx_ptr_, 
            d_activeIdx_ptr_);
        blockSizeActive = 32;
        gridSizeActive = std::min(activeSize, int(floor(maxTreeSize_ / blockSizeActive)));
        updateG_gb<<<gridSizeActive, blockSizeActive>>>(
            d_treeSamples_ptr_, 
            d_unexploredSamples_ptr_, 
            d_uParentIdx_ptr_,
            d_treeParentIdx_ptr_,
            d_G_ptr_,
            d_GNew_ptr_,
            d_activeIdx_ptr_, 
            activeSize, 
            treeSize_,
            d_costs_ptr_,
            d_xGoal_ptr_,
            goalThreshold_,
            d_costToGoal);

        
        
        treeSize_ += activeSize;

        cudaMemcpy(&costToGoal_, d_costToGoal, sizeof(float), cudaMemcpyDeviceToHost);
        if(costToGoal_ != 0){
            printf("Goal Reached\n");
            break;
        }
        if(treeSize_ >= maxTreeSize_){
            printf("Iteration %d, Tree size %d\n", itr, treeSize_);
            printf("Tree size exceeded maxTreeSize\n");
            break;
        }

        
        
        // std::ostringstream filename;
        // std::filesystem::create_directories("Data");
        // std::filesystem::create_directories("Data/Samples");
        // std::filesystem::create_directories("Data/UnexploredSamples");
        // std::filesystem::create_directories("Data/Parents");
        // std::filesystem::create_directories("Data/R1Scores");
        // std::filesystem::create_directories("Data/R1Edges");
        // std::filesystem::create_directories("Data/R1EdgeSel");
        // std::filesystem::create_directories("Data/R1EdgeVal");
        // std::filesystem::create_directories("Data/R1Avail");
        // std::filesystem::create_directories("Data/R1");
        // std::filesystem::create_directories("Data/G");
        // std::filesystem::create_directories("Data/GNew");
        // filename.str("");
        // filename << "Data/Samples/samples" << itr << ".csv";
        // copyAndWriteVectorToCSV(d_treeSamples_, filename.str(), maxTreeSize_, SAMPLE_DIM);
        // filename.str("");
        // filename << "Data/Parents/parents" << itr << ".csv";
        // copyAndWriteVectorToCSV(d_treeParentIdx_, filename.str(), maxTreeSize_, 1);
        // filename.str("");
        // filename << "Data/R1Scores/R1Scores" << itr << ".csv";
        // copyAndWriteVectorToCSV(d_R1Score_, filename.str(), N_*N_, 1);
        // filename.str("");
        // filename << "Data/R1Avail/R1Avail" << itr << ".csv";
        // copyAndWriteVectorToCSV(d_R1Avail_, filename.str(), N_*N_, 1);
        // filename.str("");
        // filename << "Data/R1/R1" << itr << ".csv";
        // copyAndWriteVectorToCSV(d_R1_, filename.str(), N_*N_, 1);
        // filename.str("");
        // filename << "Data/UnexploredSamples/unexploredSamples" << itr << ".csv";
        // copyAndWriteVectorToCSV(d_unexploredSamples_, filename.str(), maxTreeSize_, SAMPLE_DIM);
        // filename.str("");
        // filename << "Data/R1Edges/R1Edges" << itr << ".csv";
        // copyAndWriteVectorToCSV(d_R1EdgeCosts_, filename.str(), nR1Edges_, 1);
        // filename.str("");
        // filename << "Data/R1EdgeVal/R1EdgeVal" << itr << ".csv";
        // copyAndWriteVectorToCSV(d_valR1Edge_, filename.str(), nR1Edges_, 1);
        // filename.str("");
        // filename << "Data/R1EdgeSel/R1EdgeSel" << itr << ".csv";
        // copyAndWriteVectorToCSV(d_selR1Edge_, filename.str(), nR1Edges_, 1);


    }

    double t_kgmt = (std::clock() - t_kgmtStart) / (double) CLOCKS_PER_SEC;
    std::cout << "time inside GoalBiasedKGMT is " << t_kgmt << std::endl;
    printf("Iteration %d, Tree size %d\n", itr, treeSize_);

    // move vectors to csv to be plotted.
    copyAndWriteVectorToCSV(d_treeSamples_, "samples.csv", maxTreeSize_, SAMPLE_DIM);
    copyAndWriteVectorToCSV(d_unexploredSamples_, "unexploredSamples.csv", maxTreeSize_, SAMPLE_DIM);
    copyAndWriteVectorToCSV(d_treeParentIdx_, "parentRelations.csv", maxTreeSize_, 1);
    copyAndWriteVectorToCSV(d_uParentIdx_, "uParentIdx.csv", maxTreeSize_, 1);
    copyAndWriteVectorToCSV(d_G_, "G.csv", maxTreeSize_, 1);
    copyAndWriteVectorToCSV(d_R2Avail_, "R2Avail.csv", N_*N_*n_*n_, 1);
    copyAndWriteVectorToCSV(d_R1Avail_, "R1Avail.csv", N_*N_, 1);
    copyAndWriteVectorToCSV(d_R1Valid_, "R1Valid.csv", N_*N_, 1);
    copyAndWriteVectorToCSV(d_R2Valid_, "R2Valid.csv", N_*N_*n_*n_, 1);
    copyAndWriteVectorToCSV(d_R1Score_, "R1Score.csv", N_*N_, 1);
    copyAndWriteVectorToCSV(d_R1_, "R1.csv", N_*N_, 1);

    // Free the allocated memory for curand states
    cudaFree(d_randomStates);
    cudaFree(d_costToGoal);
    cudaFree(d_R1Threshold_ptr_);
}

__global__
void updateR_gb(
    float* R1Score,
    int* R2Avail,
    int* R1Avail,  
    int* R1Valid,
    int* R1,
    int n, 
    float epsilon, 
    float* R1EdgeCosts,
    int activeSize,
    int* fromR1,
    int* toR1,
    int nR1Edges,
    int* selR1Edge,
    int* valR1Edge,
    float* R1Threshold) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_alpha[NUM_R1_CELLS];
    
    if (tid >= nR1Edges) {
        return;
    }

    if (threadIdx.x < NUM_R1_CELLS) {
        int nValid = R1Valid[threadIdx.x];
        float covR = 0.0f;
        // Ensure no out-of-bounds access
        if (threadIdx.x * NUM_R2_CELLS < n && (threadIdx.x + 1) * NUM_R2_CELLS <= n) {
            for (int i = threadIdx.x * NUM_R2_CELLS; i < (threadIdx.x + 1) * NUM_R2_CELLS; i++) {
                covR += R2Avail[i];
            }
            covR /= NUM_R2_CELLS;
        }

        float freeVol = ((epsilon + nValid) / (epsilon + nValid + R1[threadIdx.x] - R1Valid[threadIdx.x]));
        s_alpha[threadIdx.x] =  1.0f / ((1 + covR) * (1 + pow(freeVol, 4)));

        if(blockIdx.x == 0){
            __shared__ float s_totalSum;
            float score = 0.0f;
            if(R1Avail[tid] != 0){
                score = pow(freeVol, 4) / ((1 + covR) * (1 + pow(R1[tid], 2)));
            }
            typedef cub::BlockReduce<float, NUM_R1*NUM_R1> BlockReduceFloatT;
            __shared__ typename BlockReduceFloatT::TempStorage tempStorageFloat;
            float blockSum = BlockReduceFloatT(tempStorageFloat).Sum(score);
            if (threadIdx.x == 0) {
                s_totalSum = blockSum;
                R1Threshold[0] = s_totalSum / activeSize;
            }
            __syncthreads();
            if(R1Avail[tid] == 0){
                R1Score[tid] = 1.0f;
            }
            else {
                R1Score[tid] = score / s_totalSum;
            }
        }

    }
    __syncthreads();

    if (tid >= nR1Edges) {
        return;
    }
    int from = fromR1[tid];
    int to = toR1[tid];
    if (from >= 0 && from < NUM_R1_CELLS && to >= 0 && to < NUM_R1_CELLS) {
        R1EdgeCosts[tid] = (1 + pow(selR1Edge[tid],2)) / (1 + pow(valR1Edge[tid],2)) * s_alpha[from] * s_alpha[to];
    }
}



// 1 Block Version. Each thread calculates 1 R1 cell.
// TODO: Change it to a 2D block. each thread square calculates 1 R1 cell. Should help with fetching R2Avail.
__global__ void updateR1_gb(
    float* R1Score, 
    int* R1Avail, 
    int* R2Avail, 
    int* R1Valid, 
    int* R1Invalid,
    int* R1,
    int n, 
    float epsilon, 
    float R1Vol,
    float* R1Threshold,
    int activeSize) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NUM_R1 * NUM_R1)
        return;

    // Use shared memory for reduction
    __shared__ float s_totalSum;

    float score = 0.0f;
    if (R1Avail[tid] != 0) {
        int nValid = R1Valid[tid];
        float covR = 0;
        for (int i = tid * n * n; i < (tid + 1) * n * n; i++) {
            covR += R2Avail[i];
        }
        covR /= n * n;

        float freeVol = ((epsilon + nValid) / (epsilon + nValid + R1Invalid[tid]));
        score = pow(freeVol, 4) / ((1 + covR) * (1 + pow(R1[tid], 2)));
    }

    typedef cub::BlockReduce<float, NUM_R1*NUM_R1> BlockReduceFloatT;
    __shared__ typename BlockReduceFloatT::TempStorage tempStorageFloat;
    float blockSum = BlockReduceFloatT(tempStorageFloat).Sum(score);

    if (threadIdx.x == 0) {
        s_totalSum = blockSum;
        R1Threshold[0] = s_totalSum / activeSize;
    }
    __syncthreads();

    // Normalize the score
    if(R1Avail[tid] == 0){
        R1Score[tid] = 1.0f;
    }
    else {
        R1Score[tid] = score / s_totalSum;
    }
}

__global__
void findInd_gb(int numSamples, bool* S, int* scanIdx, int* activeS){
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= numSamples)
        return;
    if (!S[node]) {
        return;
    }
    activeS[scanIdx[node]] = node;
}

__global__
void findInd_gb(int numSamples, int* S, int* scanIdx, int* activeS){
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= numSamples)
        return;
    if (!S[node]) {
        return;
    }
    activeS[scanIdx[node]] = node;
}

__global__ void propagateG_gb(
    int sizeG, 
    int* activeGIdx, 
    bool* G,
    bool* GNew,
    float* treeSamples,
    float* unexploredSamples,
    int* uParentIdx,
    int* R1Valid,
    int* R2Valid,
    int* R1,
    int* R2,
    int* R1Avail,
    int* R2Avail,
    int N,
    int n,
    float R1Size,
    float R2Size,
    curandState* randomStates,
    int numDisc,
    float agentLength,
    float* R1Threshold,
    float* R1Scores,
    float* obstacles,
    int obstaclesCount,
    float width,
    float height,
    int* selR1Edge,
    int* valR1Edge,
    int* hashTable,
    int tableSize) {

    // block expands x0 BLOCK_SIZE times.
    if (blockIdx.x >= sizeG)
        return;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int x0Idx;
    if(threadIdx.x == 0){
        x0Idx = activeGIdx[blockIdx.x];
        G[x0Idx] = false;
    }
    __syncthreads();
    __shared__ float x0[SAMPLE_DIM];
    if(threadIdx.x < SAMPLE_DIM){
        x0[threadIdx.x] = treeSamples[x0Idx * SAMPLE_DIM + threadIdx.x];
    }
    __syncthreads();
    curandState randState = randomStates[tid];
    float* x1 = &unexploredSamples[tid * SAMPLE_DIM];
    uParentIdx[tid] = x0Idx;
    bool valid = propagateAndCheck_gb(x0, x1, numDisc, agentLength, &randState, obstacles, obstaclesCount, width, height, selR1Edge, valR1Edge, R1Size, N, hashTable, tableSize);
    int r1 = getR1_gb(x1[0], x1[1], R1Size, N);
    int r2 = getR2_gb(x1[0], x1[1], r1, R1Size, N , R2Size, n);
    atomicAdd(&R1[r1], 1);
    atomicAdd(&R2[r2], 1);
    if(valid){
        float rand = curand_uniform(&randState);
        if(rand <= R1Scores[r1] || R2Avail[r2] == 0){
            GNew[tid] = true;
        }
        if(R1Avail[r1] == 0){
            atomicExch(&R1Avail[r1], 1);
        }
        if(R2Avail[r2] == 0){
            atomicExch(&R2Avail[r2], 1);
        }
        atomicAdd(&R2Valid[r2], 1);
        atomicAdd(&R1Valid[r1], 1);
    }
    randomStates[tid] = randState;

}
__global__ void propagateGV2_gb(
    int sizeG, 
    int* activeGIdx, 
    bool* G,
    bool* GNew,
    float* treeSamples,
    float* unexploredSamples,
    int* uParentIdx,
    int* R1Valid,
    int* R2Valid,
    int* R1,
    int* R2,
    int* R1Avail,
    int* R2Avail,
    int N,
    int n,
    float R1Size,
    float R2Size,
    curandState* randomStates,
    int numDisc,
    float agentLength,
    float* R1Threshold,
    float* R1Scores,
    float* obstacles,
    int obstaclesCount,
    int iterations,
    float width,
    float height,
    int* selR1Edge,
    int* valR1Edge){
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int x0Idx = activeGIdx[tid];
    if(!G[x0Idx]){
        return;
    }
    G[x0Idx] = false;
    int r1x0 = getR1_gb(treeSamples[x0Idx * SAMPLE_DIM], treeSamples[x0Idx * SAMPLE_DIM + 1], R1Size, N);
    float* x0 = &treeSamples[x0Idx * SAMPLE_DIM];
    for(int i = 0; i < iterations; i++){
        int x1Idx = tid * iterations + i;
        float* x1 = &unexploredSamples[x1Idx * SAMPLE_DIM];
        uParentIdx[x1Idx] = x0Idx;
        bool valid = propagateAndCheck(x0, x1, numDisc, agentLength, &randomStates[x1Idx], obstacles, obstaclesCount, width, height);
        int r1 = getR1_gb(x1[0], x1[1], R1Size, N);
        int r2 = getR2_gb(x1[0], x1[1], r1, R1Size, N, R2Size, n);
        atomicAdd(&R1[r1], 1);
        atomicAdd(&R2[r2], 1);
        if (valid) {
            float rand = curand_uniform(&randomStates[x1Idx]);
            if (rand <= R1Scores[r1] || R2Avail[r2] == 0) {
                GNew[x1Idx] = true;
            }
            if (R1Avail[r1] == 0) {
                atomicExch(&R1Avail[r1], 1);
            }
            if (R2Avail[r2] == 0) {
                atomicExch(&R2Avail[r2], 1);
            }
            atomicAdd(&R2Valid[r2], 1);
            atomicAdd(&R1Valid[r1], 1);
        }
        randomStates[x1Idx] = randomStates[x1Idx];
    }
}

__global__ void updateG_gb(
    float* treeSamples, 
    float* unexploredSamples, 
    int* uParentIdx,
    int* treeParentIdx,
    bool* G,
    bool* GNew,
    int* GNewIdx, 
    int GNewSize, 
    int treeSize,
    float* costs,
    float* xGoal,
    float r,
    float* costToGoal){
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    GNew[tid] = false;

    __shared__ float s_xGoal[SAMPLE_DIM];
    if (threadIdx.x < SAMPLE_DIM) {
        s_xGoal[threadIdx.x] = xGoal[threadIdx.x];
    }
    __syncthreads();
    
    if(tid >= GNewSize)
        return;

    // move valid unexplored sample to tree:
    int x1TreeIdx = treeSize + tid;
    int x1UnexploredIdx = GNewIdx[tid];
    float* x1 = &unexploredSamples[x1UnexploredIdx * SAMPLE_DIM];
    int x0Idx = uParentIdx[x1UnexploredIdx];
    treeParentIdx[x1TreeIdx] = x0Idx;
    treeSamples[x1TreeIdx * SAMPLE_DIM] = unexploredSamples[x1UnexploredIdx * SAMPLE_DIM];
    treeSamples[x1TreeIdx * SAMPLE_DIM + 1] = unexploredSamples[x1UnexploredIdx * SAMPLE_DIM + 1];
    treeSamples[x1TreeIdx * SAMPLE_DIM + 2] = unexploredSamples[x1UnexploredIdx * SAMPLE_DIM + 2];
    treeSamples[x1TreeIdx * SAMPLE_DIM + 3] = unexploredSamples[x1UnexploredIdx * SAMPLE_DIM + 3];
    treeSamples[x1TreeIdx * SAMPLE_DIM + 4] = unexploredSamples[x1UnexploredIdx * SAMPLE_DIM + 4];
    treeSamples[x1TreeIdx * SAMPLE_DIM + 5] = unexploredSamples[x1UnexploredIdx * SAMPLE_DIM + 5];
    treeSamples[x1TreeIdx * SAMPLE_DIM + 6] = unexploredSamples[x1UnexploredIdx * SAMPLE_DIM + 6];

    // update G:
    G[x1TreeIdx] = true;

    // update costs:
    float cost = getCost_gb(&treeSamples[x0Idx * SAMPLE_DIM], &treeSamples[x1TreeIdx * SAMPLE_DIM]);
    costs[x1TreeIdx] = costs[x0Idx] + cost;

    // check if x1 is the goal:
    if(inGoalRegion_gb(x1, s_xGoal, r)){
       costToGoal[0] = costs[x1TreeIdx];
    }
    
}

__global__ void initCurandStates_gb(curandState* states, int numStates, int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numStates)
        return;
    curand_init(seed, tid, 0, &states[tid]);
}

__device__ float getCost_gb(float* x0, float* x1){
    return x1[SAMPLE_DIM - 1]; // traj time
}

__device__ bool inGoalRegion_gb(float* x, float* goal, float r){
    float dist = sqrt(pow(x[0] - goal[0], 2) + pow(x[1] - goal[1], 2));
    return dist < r;
}