#include "sssp/sssp.cuh"
#define NUM_ASYNCHRONOUS_ITERATIONS 20  // Number of async loop iterations before attempting to read results back

#define BLOCK_SIZE 16

std::vector<int> getShortestPath(int source, int target, int* predNode) {
    std::vector<int> path;
    int currentNode = target;

    while (currentNode != source) {
        if (currentNode == -1) {
              // No path found
            return std::vector<int>();
        }
        path.push_back(currentNode);
        currentNode = predNode[currentNode];
    }
    path.push_back(source);
    std::reverse(path.begin(), path.end());
    return path;
}

__host__ void sssp(
    int numNodes, 
    int numEdges, 
    float* d_edgeCosts_ptr, 
    int *d_fromNodes_ptr, 
    int *d_toNodes_ptr, 
    float* d_dist_ptr, 
    bool* d_finished_ptr,
    int* d_predNode_ptr,
    int r1_initial,
    int r1_goal) {
    
    thrust::fill(thrust::device, d_dist_ptr, d_dist_ptr + numNodes, MAXFLOAT);
    thrust::fill(thrust::device, d_dist_ptr, d_dist_ptr + 1, 0.0f);
    int  numIteration       = 0;
    int  numEdgesPerThread  = 8;
    int  numThreadsPerBlock = 512;
    int  numBlock           = (numEdges) / (numThreadsPerBlock * numEdgesPerThread) + 1;
    bool finished           = true;
      // do {
      //     numIteration++;
      //     finished = true;
      //     cudaMemcpy(d_finished_ptr, &finished, sizeof(bool), cudaMemcpyHostToDevice);
      //     sssp_kernel<<<numBlock, numThreadsPerBlock>>>(numEdges, numEdgesPerThread, d_edgeCosts_ptr, d_fromNodes_ptr, d_toNodes_ptr, d_dist_ptr, d_finished_ptr, d_predNode_ptr);
      //     cudaMemcpy(&finished, d_finished_ptr, sizeof(bool), cudaMemcpyDeviceToHost);
      // } while (!finished && numIteration < numNodes - 1);
    int* predNode = new int[numNodes];
    cudaMemcpy(predNode, d_predNode_ptr, sizeof(int) * numNodes, cudaMemcpyDeviceToHost);
    std::vector<int> path = getShortestPath(r1_initial, r1_goal, predNode);
    if (path.empty()) {
        std::cout << "No path found from " << r1_initial << " to " << r1_goal << std::endl;
    } else {
        std::cout << "Shortest path from " << r1_initial << " to " << r1_goal << " is: ";
        for (int node : path) {
            std::cout << node << " ";
        }
        std::cout << std::endl;
    }
}

__global__ void sssp_kernel(
    int numEdges, 
    int numEdgesPerThread, 
    float* edgeCosts, 
    int *fromNodes, 
    int *toNodes, 
    float* dist,
    bool* finished,
    int* predNode) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int startId = tid * numEdgesPerThread;
    if (startId >= numEdges) {
        return;
    }
    int endId = (tid + 1) * numEdgesPerThread;
    if (endId >= numEdges) {
        endId = numEdges;
    }

    for (int nodeId = startId; nodeId < endId; nodeId++) {
        int fromNode = fromNodes[nodeId];
        int toNode = toNodes[nodeId];
        float edgeCost = edgeCosts[nodeId];
        if(dist[fromNode] + edgeCost < dist[toNode]) {
            atomicMinFloat(&dist[toNode], dist[fromNode] + edgeCost);
            predNode[toNode] = fromNode;
            *finished = false;
        }
    }
}

/**************************/
/* DIJKSTRA GPU KERNEL #1 */
/**************************/
__global__  void Kernel1(const int * __restrict__ vertexArray, const int* __restrict__ edgeArray,
                         const float * __restrict__ weightArray, bool * __restrict__ finalizedVertices, float* __restrict__ shortestDistances,
                         float * __restrict__ updatingShortestDistances, const int numVertices, const int numEdges) {

    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    if (tid < numVertices) {

        if (finalizedVertices[tid] == true) {

            finalizedVertices[tid] = false;

            int edgeStart = vertexArray[tid], edgeEnd;

            if (tid + 1 < (numVertices)) edgeEnd = vertexArray[tid + 1];
            else                         edgeEnd = numEdges;

            for (int edge = edgeStart; edge < edgeEnd; edge++) {
                int nid = edgeArray[edge];
                atomicMinFloat(&updatingShortestDistances[nid], shortestDistances[tid] + weightArray[edge]);
            }
        }
    }
}

/***********************/
/* GRAPHDATA STRUCTURE */
/***********************/
// --- The graph data structure is an adjacency list.
typedef struct {

    // --- Contains the integer offset to point to the edge list for each vertex
    int *vertexArray;

    // --- Overall number of vertices
    int numVertices;

    // --- Contains the "destination" vertices each edge is attached to
    int *edgeArray;

    // --- Overall number of edges
    int numEdges;

    // --- Contains the weight of each edge
    float *weightArray;

} GraphData;

/**************************/
/* DIJKSTRA GPU KERNEL #1 */
/**************************/
__global__  void Kernel2(const int * __restrict__ vertexArray, const int * __restrict__ edgeArray, const float* __restrict__ weightArray,
                         bool * __restrict__ finalizedVertices, float* __restrict__ shortestDistances, float* __restrict__ updatingShortestDistances,
                         const int numVertices) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numVertices) {

        if (shortestDistances[tid] > updatingShortestDistances[tid]) {
            shortestDistances[tid] = updatingShortestDistances[tid];
            finalizedVertices[tid] = true; }

        updatingShortestDistances[tid] = shortestDistances[tid];
    }
}

/*************************/
/* ARRAY INITIALIZATIONS */
/*************************/
__global__ void initializeArrays(bool * __restrict__ d_finalizedVertices, float* __restrict__ d_shortestDistances, float* __restrict__ d_updatingShortestDistances,
                                 const int sourceVertex, const int numVertices) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numVertices) {

        if (sourceVertex == tid) {

            d_finalizedVertices[tid]            = true;
            d_shortestDistances[tid]            = 0.f;
            d_updatingShortestDistances[tid]    = 0.f; }

        else {

            d_finalizedVertices[tid]            = false;
            d_shortestDistances[tid]            = MAXFLOAT;
            d_updatingShortestDistances[tid]    = MAXFLOAT;
        }
    }
}

/***************************/
/* MASKARRAYEMPTY FUNCTION */
/***************************/
// --- Check whether all the vertices have been finalized. This tells the algorithm whether it needs to continue running or not.
bool allFinalizedVertices(bool *finalizedVertices, int numVertices) {

    for (int i = 0; i < numVertices; i++)  if (finalizedVertices[i] == true) { return false; }

    return true;
}

inline int iDivUp(int a, int b) {
    return (a + b - 1) / b;
}

/************************/
/* dijkstraGPU FUNCTION */
/************************/
__host__ void dijkstraGPU(GraphData *graph, const int sourceVertex, float * __restrict__ h_shortestDistances, const int numVertices){


    // --- Create device-side adjacency-list, namely, vertex array Va, edge array Ea and weight array Wa from G(V,E,W)
    int     *d_vertexArray;         cudaMalloc(&d_vertexArray,    sizeof(int)   * graph->numVertices);
    int     *d_edgeArray;           cudaMalloc(&d_edgeArray,  sizeof(int)   * graph -> numEdges);
    float   *d_weightArray;         cudaMalloc(&d_weightArray,    sizeof(float) * graph -> numEdges);

    // --- Copy adjacency-list to the device
    cudaMemcpy(d_vertexArray, graph -> vertexArray, sizeof(int)   * graph -> numVertices, cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeArray,   graph -> edgeArray,   sizeof(int)   * graph -> numEdges,    cudaMemcpyHostToDevice);
    cudaMemcpy(d_weightArray, graph -> weightArray, sizeof(float) * graph -> numEdges,    cudaMemcpyHostToDevice);

    // --- Create mask array Ma, cost array Ca and updating cost array Ua of size V
    bool    *d_finalizedVertices;           cudaMalloc(&d_finalizedVertices,       sizeof(bool)   * graph->numVertices);
    float   *d_shortestDistances;           cudaMalloc(&d_shortestDistances,       sizeof(float) * graph->numVertices);
    float   *d_updatingShortestDistances;   cudaMalloc(&d_updatingShortestDistances, sizeof(float) * graph->numVertices);

    bool *h_finalizedVertices = (bool *)malloc(sizeof(bool) * graph->numVertices);

    // --- Initialize mask Ma to false, cost array Ca and Updating cost array Ua to \u221e
    initializeArrays <<<iDivUp(graph->numVertices, BLOCK_SIZE), BLOCK_SIZE >>>(d_finalizedVertices, d_shortestDistances,
                                                            d_updatingShortestDistances, sourceVertex, graph -> numVertices);
    cudaPeekAtLastError();
    cudaDeviceSynchronize();

    // --- Read mask array from device -> host
    cudaMemcpy(h_finalizedVertices, d_finalizedVertices, sizeof(bool) * graph->numVertices, cudaMemcpyDeviceToHost);

    while (!allFinalizedVertices(h_finalizedVertices, numVertices)) {

        // --- In order to improve performance, we run some number of iterations without reading the results.  This might result
        //     in running more iterations than necessary at times, but it will in most cases be faster because we are doing less
        //     stalling of the GPU waiting for results.
        for (int asyncIter = 0; asyncIter < NUM_ASYNCHRONOUS_ITERATIONS; asyncIter++) {

            Kernel1 <<<iDivUp(graph->numVertices, BLOCK_SIZE), BLOCK_SIZE >>>(d_vertexArray, d_edgeArray, d_weightArray, d_finalizedVertices, d_shortestDistances,
                                                            d_updatingShortestDistances, graph->numVertices, graph->numEdges);
            cudaPeekAtLastError();
            cudaDeviceSynchronize();
            Kernel2 <<<iDivUp(graph->numVertices, BLOCK_SIZE), BLOCK_SIZE >>>(d_vertexArray, d_edgeArray, d_weightArray, d_finalizedVertices, d_shortestDistances, d_updatingShortestDistances,
                                                            graph->numVertices);
            cudaPeekAtLastError();
            cudaDeviceSynchronize();
        }

        cudaMemcpy(h_finalizedVertices, d_finalizedVertices, sizeof(bool) * graph->numVertices, cudaMemcpyDeviceToHost);
    }

    // --- Copy the result to host
    cudaMemcpy(h_shortestDistances, d_shortestDistances, sizeof(float) * graph->numVertices, cudaMemcpyDeviceToHost);

    free(h_finalizedVertices);

    cudaFree(d_vertexArray);
    cudaFree(d_edgeArray);
    cudaFree(d_weightArray);
    cudaFree(d_finalizedVertices);
    cudaFree(d_shortestDistances);
    cudaFree(d_updatingShortestDistances);
    
}