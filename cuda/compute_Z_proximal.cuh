#include "compute_kernel.cuh"
#include "compute_mat_product.cuh"
/*
 * Device code
 */

template< typename T>
__global__ void kernel_Z_proximal( T* d_Zi_proximal, const T* d_Zi, const T* d_df,  const T* d_L, uint nD)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x; 
	
	if (i < nD)
	{
		d_Zi_proximal[i] = d_Zi[i] - 1.0/(*d_L)*d_df[i];
	}
}

__forceinline__ void compute_Z_proximal(float *d_Zi_proximal, float *d_DtD, float *d_DtX, float *d_Zi, float *d_L,  uint data_i, uint nD, cublasHandle_t& handle)
{
	float* d_df = NULL;
	cudaMalloc((void**)&d_df, sizeof(float)*nD);

	cudaMemcpy(d_df,d_DtX+data_i*nD,sizeof(float)*nD,cudaMemcpyDeviceToDevice);

	float ta=1.0f; float *alpha = &ta; float tb=-1.0f; float *beta = &tb;
	cublasStatus_t status = cublasSgemv(handle,
				CUBLAS_OP_N, 
				nD,           //rows of matrix A
				nD,           //cols of matrix A
				alpha,       //alpha
				d_DtD,   // A address
				nD,        // lda
				d_Zi,   // x address
				1,
				beta, 
				d_df, 1);

	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf (stderr, "!!!! kernel execution error.\n");
		return;
	}

	//compute_df(d_df, d_AtA, d_alphai, nA, handle);

	
	dim3 threadsPerBlock(MAX_THREADS_PER_BLOCK);
	dim3 blocksPerGrid((nD+threadsPerBlock.x-1)/threadsPerBlock.x);

	kernel_Z_proximal<<<blocksPerGrid, threadsPerBlock>>>(d_Zi_proximal, d_Zi, d_df, d_L, nD);

	cudaFree(d_df);
}