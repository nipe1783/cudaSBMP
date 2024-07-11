#include "sssp/sssp.cuh"



__host__ void sssp(
    int numNodes, 
    int numEdges, 
    float* d_edgeCosts_ptr, 
    int *d_fromNodes_ptr, 
    int *d_toNodes_ptr, 
    float* d_dist_ptr, 
    bool* d_finished_ptr) {
    
    thrust::fill(thrust::device, d_dist_ptr, d_dist_ptr + numNodes, MAXFLOAT);
    thrust::fill(thrust::device, d_dist_ptr, d_dist_ptr + 1, 0.0f);
    int numIteration = 0;
    int numEdgesPerThread = 8;
    int numThreadsPerBlock = 512;
    int numBlock = (numEdges) / (numThreadsPerBlock * numEdgesPerThread) + 1;
    bool finished = true;
    do {
        numIteration++;
        finished = true;
        cudaMemcpy(d_finished_ptr, &finished, sizeof(bool), cudaMemcpyHostToDevice);
        sssp_kernel<<<numBlock, numThreadsPerBlock>>>(numEdges, numEdgesPerThread, d_edgeCosts_ptr, d_fromNodes_ptr, d_toNodes_ptr, d_dist_ptr, d_finished_ptr);
        cudaMemcpy(&finished, d_finished_ptr, sizeof(bool), cudaMemcpyDeviceToHost);
    } while (!finished);
}

__global__ void sssp_kernel(
    int numEdges, 
    int numEdgesPerThread, 
    float* edgeCosts, 
    int *fromNodes, 
    int *toNodes, 
    float* dist, 
    bool* finished) {

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
            *finished = false;
        }
    }
}