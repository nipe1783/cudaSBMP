#include "sssp/sssp.cuh"


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
    int numIteration = 0;
    int numEdgesPerThread = 8;
    int numThreadsPerBlock = 512;
    int numBlock = (numEdges) / (numThreadsPerBlock * numEdgesPerThread) + 1;
    bool finished = true;
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

__global__ void kernel1(
    const int* __restrict__ fromNodes, 
    const int* __restrict__ toNodes, 
    const float* __restrict__ edgeCosts, 
    const int numEdges, 
    const int numNodes, 
    const int r1_initial, 
    const int r1_goal, 
    float* __restrict__ dist, 
    int* __restrict__ predNode, 
    bool* __restrict__ finished,
    bool* __restrict__ finalizedNodes) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < numNodes)
    {
        if(finalizedNodes[tid]) {
            finalizedNodes[tid] = false;
            int edgeStart = fromNodes[tid], edgeEnd = toNodes[tid];
            for(int edgeId = edgeStart; edgeId < edgeEnd; edgeId++) {
                atomicMinFloat(&dist[toNodes[edgeId]], dist[fromNodes[edgeId]] + edgeCosts[edgeId]);
            }
        }
    }
    

}