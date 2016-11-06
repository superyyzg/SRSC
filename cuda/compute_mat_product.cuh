#pragma once
#include "compute_kernel.cuh"

/*
 * Device code
 */

__forceinline__ void compute_AtX(float *d_AtX, float *d_A, float *d_X, uint n, uint nA, uint d, cublasHandle_t& handle)
{
	float ta=1; float *alpha = &ta; float tb=0; float *beta = &tb;
	cublasStatus_t  status = cublasSgemm (
        handle,            // blas handle 
        CUBLAS_OP_T,    //  op A
        CUBLAS_OP_N,    // op B
        nA,                // op A rows 
        n,                // op B cols
        d,                // op A cols op B rows
        alpha,                // alpha
        d_A,            // A address
        d,                // lda
        d_X,            // B address
        d,                // ldb
        beta,                // beta
        d_AtX,            // C address
        nA                // ldc
    );
	
}

//compute dst = d_A*d_alphai (alpha is sparse)
template< typename T>
__global__ void kernel_sub_A_times_sub_alphai( T* d_dst, const T* d_sub_A, const T* d_sub_alphai, 
									  uint rows, uint cols)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
		
	if (i < rows)
	{
		for (uint j = 0; j < cols; j++) //(*num_nonzero)
		{
			uint idx = j*rows+i;
			d_dst[i] = d_dst[i] + d_sub_A[idx]*d_sub_alphai[j];
		}
	}
	
}

//find positions of nonzero elements in alphai
template< typename T>
__global__ void kernel_nonzeros_alphai(T *d_sub_alphai, uint* d_pos, uint* num_nonzero, const T*d_alphai, uint n)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n)
	{
		if (abs(d_alphai[i]) > eps)
		{
			uint x = atomicAdd(num_nonzero,1);
			d_pos[x] = i;
			d_sub_alphai[x] = d_alphai[i];
		}
	}
}

//find positions of nonzero elements in alphai
template< typename T>
__global__ void kernel_sub_A(T *d_sub_A, const T *d_A, uint* d_pos, uint* num_nonzero, uint rows)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x; 
	uint j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < rows && j < (*num_nonzero))
	{
		uint idx = j*rows+i;
		d_sub_A[idx] = d_A[d_pos[j]*rows + i];
	}
}


__forceinline__ void compute_df(float *d_df, float *d_A, float *d_alphai, uint rows, cublasHandle_t& handle)
{
	uint *d_pos = NULL, *num_nonzero = NULL; 
	float *d_sub_alphai = NULL;
	cudaMalloc((void**)&d_pos, sizeof(uint)*rows);
	cudaMalloc((void**)&num_nonzero, sizeof(uint));
	cudaMalloc((void**)&d_sub_alphai, sizeof(float));

	uint *h_num_nonzero = (uint*)malloc(sizeof(uint));
	*h_num_nonzero = 0;
	cudaMemcpy(num_nonzero,h_num_nonzero,sizeof(uint),cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK);
	dim3 blocksPerGrid((rows+threadsPerBlock.x-1)/threadsPerBlock.x);
	kernel_nonzeros_alphai<<<blocksPerGrid, threadsPerBlock>>>(d_sub_alphai, d_pos, num_nonzero, d_alphai, rows);

	cudaMemcpy(h_num_nonzero,num_nonzero,sizeof(uint),cudaMemcpyDeviceToHost);
	uint sub_A_cols = *h_num_nonzero;
	float *d_sub_A = NULL;
	cudaMalloc((void**)&d_sub_A, sizeof(float)*rows*sub_A_cols);

	threadsPerBlock = dim3(BLOCK_SIZE,BLOCK_SIZE);
	blocksPerGrid = dim3( (rows+threadsPerBlock.x-1)/threadsPerBlock.x, (sub_A_cols+threadsPerBlock.y-1)/threadsPerBlock.y);
	kernel_sub_A<<<blocksPerGrid, threadsPerBlock>>>(d_sub_A, d_A, d_pos, num_nonzero, rows);
	
	//kernel_sub_A_times_sub_alphai<<<blocksPerGrid, threadsPerBlock>>>(d_dst, d_sub_A, d_sub_alphai, d_pos, num_nonzero, rows, cols);
	float ta=2.0f; float *alpha = &ta; float tb=-2.0f; float *beta = &tb;
	cublasStatus_t status = cublasSgemv(handle,
				CUBLAS_OP_N, 
				rows,           //rows of matrix A
				sub_A_cols,           //cols of matrix A
				alpha,       //alpha
				d_A,   // A address
				rows,        // lda
				d_sub_alphai,   // x address
				1,
				beta, 
				d_df, 1);

	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! kernel execution error.\n");
		return;
	}

	free(h_num_nonzero);
	cudaFree(d_pos);
	cudaFree(num_nonzero);
	cudaFree(d_sub_alphai);
	cudaFree(d_sub_A);
}