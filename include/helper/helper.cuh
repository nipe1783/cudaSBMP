#pragma once

#include <stdio.h>
#include <vector>
#include <iostream>
#include <string>
#include <ctime>
#include <cstdlib>
#include <math.h>
#include <fstream>
#include "cuda.h"
#include "cuda_runtime.h"
#define _USE_MATH_DEFINES

// A macro for checking the error codes of cuda runtime calls
#define CUDA_ERROR_CHECK(expr) \
  {                            \
    cudaError_t err = expr;    \
    if (err != cudaSuccess)    \
    {                          \
      printf("CUDA call failed!\n\t%s\n", cudaGetErrorString(err)); \
      exit(1);                 \
    }                          \
  }
