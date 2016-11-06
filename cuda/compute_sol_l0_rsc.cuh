#include "compute_kernel.cuh"

/*
 * Device code
 */

template< typename T>
__global__ void kernel_soft_threshold_l0rsc( T* dst, T* src, T *thres, uint nD)
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
void __global__ kernel_obj_mat_l0rsc(T *d_obj_mat, const T* d_all_sols, const T* d_Zi_prox, const T* d_Zi_neighbor, const T* d_Wi_neighbor, const T *d_L, const T *d_lambda, const T *d_gamma, 
							   uint nD, uint nn_size)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x; 
	uint j = blockIdx.y * blockDim.y + threadIdx.y;

	T L = *d_L;
	T lambda = *d_lambda;
	T gamma = *d_gamma;

	if (i < nD && j < (nn_size+1))
	{
		uint idx = j*nD+i;
		T val = L/2.0*(d_all_sols[idx] - d_Zi_prox[i])*(d_all_sols[idx] - d_Zi_prox[i]) + lambda*abs(d_all_sols[idx]);
		for (uint t = 0; t < nn_size; t++)
		{
			uint Z_neighbor_idx = t*nD+i;
			T l0_val = (T)(abs(d_all_sols[idx] -d_Zi_neighbor[Z_neighbor_idx]) > eps);
			val = val + gamma*d_Wi_neighbor[t]*l0_val;
		}
		d_obj_mat[idx] = val;
	}

}

template< typename T>
__global__ void kernel_Zi_from_obj_mat_l0rsc( T* d_Zi, const T* d_obj_mat, const T* d_all_sols, uint nD, uint nn_size)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;

	uint idx = 0;
	
	if (i < nD)
	{
		T row_min = d_obj_mat[i];
		uint min_col = 0;
		for (int j = 1; j < nn_size+1; j++)
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

/*
function [alphai,obj_mat] = sol_l0_l1(alphai_prox,alpha_neighbor,W0_neighbor,c,lambda_l1,lambda_l0)
    
    n = size(alphai_prox,1);
    nn = size(alpha_neighbor,2);
        
    l1_solution = max(abs(alphai_prox) - lambda_l1/c,0).*sign(alphai_prox);
    all_sols = [l1_solution alpha_neighbor];
    
    obj_mat = c/2*(all_sols - repmat(alphai_prox,1,nn+1)).^2 + lambda_l1*(abs(all_sols));
    
    for t = 1:nn,
        obj_mat = obj_mat + lambda_l0*W0_neighbor(t)*(abs(all_sols - repmat(alpha_neighbor(:,t),1,nn+1))>eps);
    end
    
    [~,min_idx] = min(obj_mat,[],2);
    
    alphai = all_sols(sub2ind([n nn+1],(1:n)',min_idx));
end
*/

__forceinline__ void compute_sol_l0_rsc(float* d_Zi, float* d_Zi_prox, float* d_Zi_neighbor, float* d_Wi_neighbor, uint data_i, uint nD, uint nn_size, float *d_L, float *d_lambda, float *d_lambda_L,
									   float *d_gamma)
{
	float* d_l1_solution = NULL, *d_all_sols = NULL, *d_obj_mat = NULL;
	cudaMalloc((void**)&d_l1_solution, sizeof(float)*nD);
	cudaMalloc((void**)&d_all_sols, sizeof(float)*nD*(nn_size+1));
	cudaMalloc((void**)&d_obj_mat, sizeof(float)*nD*(nn_size+1));
	
	dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK);
	dim3 blocksPerGrid( (nD+threadsPerBlock.x-1)/threadsPerBlock.x);
	
	kernel_soft_threshold_l0rsc<<<blocksPerGrid, threadsPerBlock>>>(d_l1_solution, d_Zi_prox, d_lambda_L, nD);

	cudaMemcpy(d_all_sols,d_l1_solution,sizeof(float)*nD,cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_all_sols+nD,d_Zi_neighbor,sizeof(float)*nD*nn_size,cudaMemcpyDeviceToDevice);

	threadsPerBlock = dim3(BLOCK_SIZE,BLOCK_SIZE);
	blocksPerGrid = dim3((nD+threadsPerBlock.x-1)/threadsPerBlock.x,(nn_size+1+threadsPerBlock.y-1)/threadsPerBlock.y);
	kernel_obj_mat_l0rsc<<<blocksPerGrid, threadsPerBlock>>>(d_obj_mat, d_all_sols, d_Zi_prox, d_Zi_neighbor, d_Wi_neighbor, d_L, d_lambda, d_gamma, nD, nn_size);

	kernel_Zi_from_obj_mat_l0rsc<<<blocksPerGrid, threadsPerBlock>>>(d_Zi, d_obj_mat, d_all_sols, nD, nn_size);

	cudaFree(d_l1_solution);
	cudaFree(d_all_sols);
	cudaFree(d_obj_mat);
}