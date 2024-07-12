#include "helper/helper.cuh"
#include "sssp/sssp.cuh"
#define NUM_ASYNCHRONOUS_ITERATIONS 20  // Number of async loop iterations before attempting to read results back
#define BLOCK_SIZE 16

std::vector<int> getShortestPath(int source, int target, int* predNode) {
    std::vector<int> path;
    int currentNode = target;
    while(currentNode != source)
        {
            if(currentNode == -1)
                {
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

__host__ void sssp(int numNodes, int numEdges, float* d_edgeCosts_ptr, int* d_fromNodes_ptr, int* d_toNodes_ptr, float* d_dist_ptr,
                   bool* d_finished_ptr, int* d_predNode_ptr, int r1_initial, int r1_goal) {
    thrust::fill(thrust::device, d_dist_ptr, d_dist_ptr + numNodes, MAXFLOAT);
    thrust::fill(thrust::device, d_dist_ptr, d_dist_ptr + 1, 0.0f);
    int numIteration = 0;
    int numEdgesPerThread = 8;
    int numThreadsPerBlock = 512;
    int numBlock = (numEdges) / (numThreadsPerBlock * numEdgesPerThread) + 1;
    bool finished = true;
    // do {
    //     numIteration++;
    //     finished = true;
    //     cudaMemcpy(d_finished_ptr, &finished, sizeof(bool),
    //     cudaMemcpyHostToDevice); sssp_kernel<<<numBlock,
    //     numThreadsPerBlock>>>(numEdges, numEdgesPerThread, d_edgeCosts_ptr,
    //     d_fromNodes_ptr, d_toNodes_ptr, d_dist_ptr, d_finished_ptr,
    //     d_predNode_ptr); cudaMemcpy(&finished, d_finished_ptr, sizeof(bool),
    //     cudaMemcpyDeviceToHost);
    // } while (!finished && numIteration < numNodes - 1);
    int* predNode = new int[numNodes];
    cudaMemcpy(predNode, d_predNode_ptr, sizeof(int) * numNodes, cudaMemcpyDeviceToHost);
    std::vector<int> path = getShortestPath(r1_initial, r1_goal, predNode);
    if(path.empty())
        {
            std::cout << "No path found from " << r1_initial << " to " << r1_goal << std::endl;
        }
    else
        {
            std::cout << "Shortest path from " << r1_initial << " to " << r1_goal << " is: ";
            for(int node : path)
                {
                    std::cout << node << " ";
                }
            std::cout << std::endl;
        }
}

__global__ void
sssp_kernel(int numEdges, int numEdgesPerThread, float* edgeCosts, int* fromNodes, int* toNodes, float* dist, bool* finished, int* predNode) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int startId = tid * numEdgesPerThread;
    if(startId >= numEdges)
        {
            return;
        }
    int endId = (tid + 1) * numEdgesPerThread;
    if(endId >= numEdges)
        {
            endId = numEdges;
        }
    for(int nodeId = startId; nodeId < endId; nodeId++)
        {
            int fromNode = fromNodes[nodeId];
            int toNode = toNodes[nodeId];
            float edgeCost = edgeCosts[nodeId];
            if(dist[fromNode] + edgeCost < dist[toNode])
                {
                    atomicMinFloat(&dist[toNode], dist[fromNode] + edgeCost);
                    predNode[toNode] = fromNode;
                    *finished = false;
                }
        }
}

/**************************/
/* DIJKSTRA GPU KERNEL #1 */
/**************************/
__global__ void Kernel1(const int* __restrict__ vertexArray, const int* __restrict__ edgeArray, const float* __restrict__ weightArray,
                        bool* __restrict__ finalizedVertices, float* __restrict__ shortestDistances, float* __restrict__ updatingShortestDistances,
                        const int numVertices, const int numEdges) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < numVertices)
        {
            if(finalizedVertices[tid] == true)
                {
                    finalizedVertices[tid] = false;
                    int edgeStart = vertexArray[tid], edgeEnd;
                    if(tid + 1 < (numVertices))
                        edgeEnd = vertexArray[tid + 1];
                    else
                        edgeEnd = numEdges;
                    for(int edge = edgeStart; edge < edgeEnd; edge++)
                        {
                            int nid = edgeArray[edge];
                            atomicMinFloat(&updatingShortestDistances[nid], shortestDistances[tid] + weightArray[edge]);
                        }
                }
        }
}

/**************************/
/* DIJKSTRA GPU KERNEL #1 */
/**************************/
__global__ void Kernel2(const int* __restrict__ vertexArray, const int* __restrict__ edgeArray, const float* __restrict__ weightArray,
                        bool* __restrict__ finalizedVertices, float* __restrict__ shortestDistances, float* __restrict__ updatingShortestDistances,
                        const int numVertices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < numVertices)
        {
            if(shortestDistances[tid] > updatingShortestDistances[tid])
                {
                    shortestDistances[tid] = updatingShortestDistances[tid];
                    finalizedVertices[tid] = true;
                }
            updatingShortestDistances[tid] = shortestDistances[tid];
        }
}

/***************************/
/* MASKARRAYEMPTY FUNCTION */
/***************************/
// --- Check whether all the vertices have been finalized. This tells the
// algorithm whether it needs to continue running or not.
bool allFinalizedVertices(bool* finalizedVertices, int numVertices) {
    for(int i = 0; i < numVertices; i++)
        if(finalizedVertices[i] == true)
            {
                return false;
            }
    return true;
}

inline int iDivUp(int a, int b) {
    return (a + b - 1) / b;
}

/************************/
/* dijkstraGPU FUNCTION */
/************************/
__host__ void
dijkstraGPU(const int sourceVertex, const int numVertices, const int numEdges, int* d_vertexArray, int* d_edgeArray, float* d_weightArray,
            bool* d_finalizedVertices, float* d_shortestDistances, float* d_updatingShortestDistances, bool* h_finalizedVertices) {
    
    // --- Initialize Arrays
    thrust::fill(thrust::device, d_shortestDistances, d_shortestDistances + numVertices, MAXFLOAT);
    thrust::fill(thrust::device, d_shortestDistances + sourceVertex, d_shortestDistances + sourceVertex + 1, 0.0f);
    thrust::fill(thrust::device, d_updatingShortestDistances, d_updatingShortestDistances + numVertices, MAXFLOAT);
    thrust::fill(thrust::device, d_updatingShortestDistances + sourceVertex, d_updatingShortestDistances + sourceVertex + 1, 0.0f);
    thrust::fill(thrust::device, d_finalizedVertices, d_finalizedVertices + numVertices, false);
    thrust::fill(thrust::device, d_finalizedVertices + sourceVertex, d_finalizedVertices + sourceVertex + 1, true);

    // --- Read mask array from device -> host
    cudaMemcpy(h_finalizedVertices, d_finalizedVertices, sizeof(bool) * numVertices, cudaMemcpyDeviceToHost);

    while(!allFinalizedVertices(h_finalizedVertices, numVertices))
        {
            // --- In order to improve performance, we run some number of iterations
            // without reading the results.  This might result
            //     in running more iterations than necessary at times, but it will in
            //     most cases be faster because we are doing less stalling of the GPU
            //     waiting for results.
            for(int asyncIter = 0; asyncIter < NUM_ASYNCHRONOUS_ITERATIONS; asyncIter++)
                {
                    Kernel1<<<iDivUp(numVertices, BLOCK_SIZE), BLOCK_SIZE>>>(d_vertexArray, d_edgeArray, d_weightArray, d_finalizedVertices,
                                                                             d_shortestDistances, d_updatingShortestDistances, numVertices, numEdges);
                    cudaPeekAtLastError();
                    cudaDeviceSynchronize();
                    Kernel2<<<iDivUp(numVertices, BLOCK_SIZE), BLOCK_SIZE>>>(d_vertexArray, d_edgeArray, d_weightArray, d_finalizedVertices,
                                                                             d_shortestDistances, d_updatingShortestDistances, numVertices);
                    cudaPeekAtLastError();
                    cudaDeviceSynchronize();
                }
            cudaMemcpy(h_finalizedVertices, d_finalizedVertices, sizeof(bool) * numVertices, cudaMemcpyDeviceToHost);
        }
}
