#include "compute_kernel.cuh"

/*
 * Device code
 */

template< typename T>
__global__ void kernel_X_A_alphai( T* dst, const T* d_X, const T* d_A_alphai, uint data_i, uint d)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x; 
	
	if (i < d)
	{
		dst[i]  = d_X[data_i*d+i] - d_A_alphai[i];
	}
}

template< typename T>
__global__ void kernel_l0_spar_err( T* dst, const T* d_alphai, const T* d_alphai_neighbor, const T* d_W0_neighbor, const T *d_lambda_l0, uint nA, uint nn)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x; 
	uint j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < nA && j < nn)
	{
		uint idx = j*nA+i;
		dst[idx]  = (*d_lambda_l0)*d_W0_neighbor[j]*(T)(abs(d_alphai[i] - d_alphai_neighbor[idx]) > eps);
	}
}


/*
function [obj,l2err,l1_spar_err,l0_spar_err] = compute_obj_robust(X,i,A,alphai,alpha_neighbor,W0_neighbor,lambda_l1,lambda_l0)
    n = size(alphai,1);
    l2err = norm(X(:,i)-A*alphai,'fro')^2;
    l1_spar_err = lambda_l1*sum(abs(alphai));
    
    nn = size(alpha_neighbor,2);
    l0_spar_err = lambda_l0*(abs(repmat(alphai,1,nn) - alpha_neighbor)>eps).*repmat(W0_neighbor,n,1);
    l0_spar_err = sum(l0_spar_err(:));
    obj = l2err + l1_spar_err + l0_spar_err;
end
*/

__forceinline__ void compute_obj_robust(float& obj, float& l2err, float& l1_spar_err, float& l0_spar_err, float *d_X, uint data_i, float *d_A, float *d_alphai, float *d_alphai_neighbor, 
									   float *d_W0_neighbor, float lambda_l1, float *d_lambda_l0, uint n, uint nA, uint nn, uint d, cublasHandle_t& handle)
{
	float* d_X_A_alphai = NULL, *d_l0_spar_err_mat = NULL;
	cudaMalloc((void**)&d_X_A_alphai, sizeof(float)*d);
	cudaMalloc((void**)&d_l0_spar_err_mat, sizeof(float)*n*nn);

	float ta=1; float *alpha = &ta; float tb=0; float *beta = &tb;
	cublasStatus_t status = cublasSgemv(handle,
				CUBLAS_OP_N, 
				d,           //rows of matrix A
				nA,           //cols of matrix A
				alpha,       //alpha
				d_A,   // A address
				d,        // lda
				d_alphai,   // x address
				1,
				beta, 
				d_X_A_alphai, 1);

	dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK);
	dim3 blocksPerGrid( (d+threadsPerBlock.x-1)/threadsPerBlock.x);

	kernel_X_A_alphai<<<blocksPerGrid, threadsPerBlock>>>(d_X_A_alphai, d_X, d_X_A_alphai, data_i, d);

	status = cublasSnrm2(handle, d, d_X_A_alphai, 1, &l2err);
	l2err = l2err*l2err;
	//debug
	//printf("l2err1 = %.5f \n", l2err);

	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! kernel execution error.\n");
		return;
	}

	status = cublasSasum(handle, nA, d_alphai, 1, &l1_spar_err);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! kernel execution error.\n");
		return;
	}

	//cublasSscal(handle, 1, d_lambda_l1, l1_spar_err, 1);
	l1_spar_err = lambda_l1*l1_spar_err;

	threadsPerBlock = dim3(BLOCK_SIZE,BLOCK_SIZE);
	blocksPerGrid = dim3((nA+threadsPerBlock.x-1)/threadsPerBlock.x,(nn+threadsPerBlock.y-1)/threadsPerBlock.y);
	kernel_l0_spar_err<<<blocksPerGrid, threadsPerBlock>>>(d_l0_spar_err_mat, d_alphai, d_alphai_neighbor, d_W0_neighbor, d_lambda_l0, nA, nn);

	status = cublasSasum(handle, nA*nn, d_l0_spar_err_mat, 1, &l0_spar_err);
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! kernel execution error.\n");
		return;
	}

	obj = l2err + l1_spar_err + l0_spar_err;

	cudaFree(d_X_A_alphai);
	cudaFree(d_l0_spar_err_mat);

}