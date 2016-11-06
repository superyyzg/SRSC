#include "compute_kernel.cuh"

/*
 * Device code
 */

template< typename T>
__global__ void kernel_neighbor_size( uint* d_neighbor_size, const T* d_adjmat, uint adjmat_size, uint n)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x; 
	uint j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < adjmat_size && j < n)
	{
		uint idx = j*adjmat_size+i;
		if (d_adjmat[idx] > 0)
			atomicAdd(d_neighbor_size+j, 1);

	}
}

template< typename T>
__global__ void kernel_Z_neighbor_weight( T* d_Zi_neighbor, T *d_Wi_neighbor, const T* d_adjmat, const T* d_adjweight, const T* d_Z, uint data_i, uint nD, uint nn_size, uint adjmat_size)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x; 
	uint j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < nD && j < nn_size)
	{
		uint idx = j*nD+i;
		d_Zi_neighbor[idx] = d_Z[(uint)(d_adjmat[data_i*adjmat_size+j]-1)*nD + i];
		d_Wi_neighbor[j] = d_adjweight[data_i*adjmat_size+j];
	}
}

__forceinline__ void compute_neighbor_size(uint* d_neighbor_size, float *d_adjmat, uint adjmat_size, uint n)
{
	dim3 threadsPerBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 blocksPerGrid((adjmat_size+threadsPerBlock.x-1)/threadsPerBlock.x,(n+threadsPerBlock.y-1)/threadsPerBlock.y);

	kernel_neighbor_size<<<blocksPerGrid, threadsPerBlock>>>(d_neighbor_size, d_adjmat, adjmat_size, n);
}

// d_adjmat: size nn x n
__forceinline__ void compute_Zi_neighbor(float *d_Zi_neighbor, float *d_Wi_neighbor, float *d_adjmat, float *d_adjweight, float *d_Z, uint data_i, uint nD, uint nn_size, uint adjmat_size)
{
	dim3 threadsPerBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 blocksPerGrid((nD+threadsPerBlock.x-1)/threadsPerBlock.x,(nn_size+threadsPerBlock.y-1)/threadsPerBlock.y);

	kernel_Z_neighbor_weight<<<blocksPerGrid, threadsPerBlock>>>(d_Zi_neighbor, d_Wi_neighbor, d_adjmat, d_adjweight, d_Z, data_i, nD, nn_size, adjmat_size);
}