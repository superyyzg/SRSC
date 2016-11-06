#include "compute_kernel.cuh"
#include "utility.h"

/*
 * Device code
 */

#define SUPPORT_DIFF(a,b) (abs((a)-(b)) > eps) && ((abs(a) < eps) || (abs(b) < eps))

template< typename T>
__global__ void kernel_soft_threshold( T* dst, T* src, T *thres, uint nD)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x; 
	
	if (i < nD)
	{
		T st = abs(src[i]) - (*thres);

		T src_sgn = (T)((src[i] > (T)0) - ((src[i] < (T)0)));
		
		if (st < (T)0)
			st = (T)0;

		dst[i]  = st*src_sgn;		
	}
}

template< typename T>
void __global__ kernel_obj_mat(T *d_obj_mat, const T* d_all_sols, const T* d_Zi_prox, const T* d_Zi_neighbor, const T* d_Wi_neighbor, const T *d_L, const T *d_lambda, const T *d_gamma, 
							   uint nD, uint nn_size, uint sol_size)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x; 
	uint j = blockIdx.y * blockDim.y + threadIdx.y;

	T L = *d_L;
	T lambda = *d_lambda;
	T gamma = *d_gamma;

	if (i < nD && j < sol_size)
	{
		uint idx = j*nD+i;
		T val = L/2.0*(d_all_sols[idx] - d_Zi_prox[i])*(d_all_sols[idx] - d_Zi_prox[i]) + lambda*abs(d_all_sols[idx]);
		for (uint t = 0; t < nn_size; t++)
		{
			uint Z_neighbor_idx = t*nD+i;
			T supp = (T) SUPPORT_DIFF(d_all_sols[idx], d_Zi_neighbor[Z_neighbor_idx]);
			val = val + gamma*d_Wi_neighbor[t]*supp;
		}
		d_obj_mat[idx] = val;
	}

}

template< typename T>
__global__ void kernel_Zi_from_obj_mat( T* d_Zi, const T* d_obj_mat, const T* d_all_sols, uint nD, uint sol_size)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;

	uint idx = 0;
	
	if (i < nD)
	{
		T row_min = d_obj_mat[i];
		uint min_col = 0;
		for (int j = 1; j < sol_size; j++)
		{
			idx = j*nD+i;
			if (d_obj_mat[idx] < row_min)
			{
				row_min = d_obj_mat[idx];
				min_col = j;
			}
		}
		d_Zi[i] = d_all_sols[min_col*nD+i];
	}
}

__forceinline__ void compute_sol_srsc(float* d_Zi, float* d_Zi_prox, float* d_Zi_neighbor, float* d_Wi_neighbor, uint data_i, uint nD, uint nn_size, float *d_L, float *d_lambda, float *d_lambda_L,
									   float *d_gamma)
{
	float* d_l1_solution = NULL, *d_all_sols = NULL, *d_obj_mat = NULL;
	cudaMalloc((void**)&d_l1_solution, sizeof(float)*nD);
	const int sol_size = 2;
	cudaMalloc((void**)&d_all_sols, sizeof(float)*nD*sol_size);
	cudaMalloc((void**)&d_obj_mat, sizeof(float)*nD*sol_size);

	dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK);
	dim3 blocksPerGrid( (nD+threadsPerBlock.x-1)/threadsPerBlock.x);
	
	kernel_soft_threshold<<<blocksPerGrid, threadsPerBlock>>>(d_l1_solution, d_Zi_prox, d_lambda_L, nD);

	cudaMemcpy(d_all_sols,d_l1_solution,sizeof(float)*nD,cudaMemcpyDeviceToDevice);
	//cudaMemcpy(d_all_sols+nA,d_alpha_neighbor,sizeof(float)*nA*nn_size,cudaMemcpyDeviceToDevice);
	cudaMemset(d_all_sols+nD,0,sizeof(float)*nD);

	threadsPerBlock = dim3(BLOCK_SIZE,BLOCK_SIZE);
	blocksPerGrid = dim3((nD+threadsPerBlock.x-1)/threadsPerBlock.x,(sol_size+threadsPerBlock.y-1)/threadsPerBlock.y);
	kernel_obj_mat<<<blocksPerGrid, threadsPerBlock>>>(d_obj_mat, d_all_sols, d_Zi_prox, d_Zi_neighbor, d_Wi_neighbor, d_L, d_lambda, d_gamma, nD, nn_size, sol_size);

	//debug
	//if (data_i == 105)
	//{
	//	export_variable(d_obj_mat, "d_obj_mat", nD, sol_size);
	//	export_variable(d_Wi_neighbor, "d_Wi_neighbor", nn_size, 1);
	//	export_variable(d_l1_solution, "d_l1_solution", nD, 1);
	//	export_variable(d_supp_mat, "d_supp_mat", nD, 2);
	//}

	kernel_Zi_from_obj_mat<<<blocksPerGrid, threadsPerBlock>>>(d_Zi, d_obj_mat, d_all_sols, nD, sol_size);

	cudaFree(d_l1_solution);
	cudaFree(d_all_sols);
	cudaFree(d_obj_mat);
	
}