#pragma once

#include "planners/Planner.cuh"
#include "occupancyMaps/OccupancyGrid.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <curand_kernel.h>

class KGMT
{
    public:
        // constructor
        KGMT() = default;
        KGMT(float width, float height, int N, int numIterations, int numSamples, int numDisc, int sampleDim, float agentLength);

        // methods
        void plan(float* initial, float* goal);

        // fields
        int numIterations_; // Number of iterations to run KGMT
        int maxSamples_; // Maximum number of samples to store. Similar to number of samples in PRM exceept it is initialized to 0s.
        int numDisc_; // Number of iterations when doing state propagation
        int N_; // Number of cols/rows in the workspace grid
        int sampleDim_; // Dimension of each sample
        int treeSize_; // Current size of the tree. Informs where in the d_samples_ vector to add new samples.
        float width_; // Width of the workspace
        float height_; // Height of the workspace
        float cellSize_; // Size of each cell in the workspace grid
        float costGoal_; // Cost of the goal state
        float agentLength_; // Length of the agent. Used in state propagation.
        float connThresh_;
        OccupancyGrid grid_; // Workspace grid.
        thrust::device_vector<bool> d_eOpen_; // True if sample is part of open samples to possibly be added to G.
        thrust::device_vector<bool> d_eClosed_; // Already been expanded by G or cost is too high.
        thrust::device_vector<bool> d_G_; // Set of samples to be expanded in current iteration.
        thrust::device_vector<bool> d_edges_; // TODO: Delete this. Not used.
        thrust::device_vector<bool> d_eUnexplored_; // Set of samples created by an iteration of G. To be added to vOpen or discarded.
        thrust::device_vector<int> d_scanIdx_; // stores scan of G. ex: G = [0, 1, 0, 1, 1, 0, 1] -> scanIdx = [0,0,1,1,2,3,3]. Used to find active samples in G.
        thrust::device_vector<int> d_activeGIdx_; // Used to store indeces that are true in G. ex [1,3,5] means samples 1,3,5 are in G.
        thrust::device_vector<int> d_activeIdx_;
        thrust::device_vector<int> d_eParentIdx_; // Stores parent sample idx. Ex: d_parentIdx[10] = 3. Parent of sample 3 is 10.
        thrust::device_vector<float> d_eConnectivity_; // Stores connectivty score for each sample. 
        thrust::device_vector<float> d_samples_; // Stores all samples. Size is maxSamples_ * sampleDim_.
        bool *d_eOpen_ptr_;
        bool *d_eClosed_ptr_;
        bool *d_G_ptr_;
        bool *d_edges_ptr_;
        bool *d_eUnexplored_ptr_;
        float *d_samples_ptr_;
        float *d_costGoal;
        float *d_eConnectivity_ptr_;
        int *d_scanIdx_ptr_;
        int *d_activeIdx_G_ptr_; // TODO: Possibly delete.
        int *d_activeIdx_ptr_;
        int *d_eParentIdx_ptr_;
        
};

// GPU kernels:

/**
 * @brief Populates activeGIdx with the indeces of the samples in G.
 *  EX: G = [0, 1, 0, 1, 1, 0, 1] -> activeGIdx = [1,3,4,6,0,0,0]
 */
__global__ void findInd(int numSamples, bool* G, int* scanIdx, int* activeGIdx);

/**
 * @brief Expands each sample in G. 
 * Adds new samples to eUnexplored. Calculates connectivity score for new samples. 
 * Moves G samples to eClosed.
 */
__global__ void propagateG(float* samples, bool* eUnexplored, bool* eClosed, bool* G, float* eConn, int* eParentIDx, int* activeIdx_G, int activeSize_G, int treeSize, int sampleDim, float agentLength, int numDisc, curandState* states);
__global__ void expandEOpen(bool* eUnexplored, bool* eClosed, bool* eOpen, float* eConn, int* activeEUnexplored_Idx, int size_activeEUnexplored, float connThresh);
__global__ void expandG(bool* eOpen, bool* G, float* eConn, int* activeEOpen_Idx, int size_activeEOpen, float connThresh);
__global__ void initCurandStates(curandState* states, int numStates, int seed);

// GPU device functions:

/**
 * @brief Populates X1 with new sample. 
 *
 */
__device__ void propagateState(float* x0, float* x1, int numDisc, float agentLength, curandState* state);

/**
 * @brief Calculates connectivity score for a sample.
 *
 */
__device__ float calculateConnectivity(float* x, curandState* state);
