#pragma once
#include<iostream>
#include <fstream>
#include "mat.h"
#include<string>

#define _MY_DEBUG


typedef unsigned int uint;

namespace SrSCIO
{
	void process_input(float*& X, float*& l1graph_alpha, float*& adjmat, float*& A, float*& AtA, float& S1, uint& n, uint& d, uint &nn, const char *input_matfile_name);
}

#ifdef _MY_DEBUG
//debug function
#include "cuda_runtime.h"
#include <cuda.h>
#include "cublas_v2.h"
void export_variable(const float *d_data, const char *name, uint h, uint w);
#endif
