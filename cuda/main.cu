//#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

#include "compute_kernel.cuh"
#include "compute_obj_robust.cuh"
#include "compute_sol_srsc.cuh"
#include "compute_sol_l0_rsc.cuh"
#include "compute_Zi_neighbor.cuh"
#include "compute_mat_product.cuh"
#include "compute_Z_proximal.cuh"
#include "compute_error_coef.cuh"
#include "utility.h"





/*
for outer_iter = 1:maxIter,


for i = 1:n,
    
    alpha_neighbor = alpha(:,adjmat(:,i));
    W0_neighbor = ones(1,knn);
    
	alphai0 = alpha(:,i); 
    alphai = alphai0;
    
    %[obj,l2err,l1_spar_err,l0_spar_err] = compute_obj_robust(X,i,A,alphai0,alpha_neighbor,W0_neighbor,lambda_l1,lambda_l0);
    
    %fprintf('obj is %.5f, l2err is %.5f, l1_spar_err is %.5f l0_spar_err is %.5f\n', obj,l2err,l1_spar_err,l0_spar_err);
    
    iter = 1;
    while ( iter <= maxSingleIter )
        %add robustness to noise and outlier
        df = 2*(AtA*alphai-AtX(:,i));
        c = 2*S(1);
        
        alpha_proximal = alphai - 1/c*df;

        [alphai,~] = sol_l0_l1(alpha_proximal,alpha_neighbor,W0_neighbor,c,lambda_l1,lambda_l0);

        alphai(i) = 0;
        
		err = errorCoef(alphai,alphai0);
		
		alphai0 = alphai;

        [obj,l2err,l1_spar_err,l0_spar_err] = compute_obj_robust(X,i,A,alphai,alpha_neighbor,W0_neighbor,lambda_l1,lambda_l0);
        
        if verbose,
            fprintf('proximal_manifold: errors = [%1.1e], iter: %4.0f \n',err,iter);
            fprintf('obj is %.5f, l2err is %.5f, l1_spar_err is %.5f l0_spar_err is %.5f\n', obj,l2err,l1_spar_err,l0_spar_err);
        end

        iter = iter+1;

    end %while
    
    alpha(:,i) = alphai;
    lastprintlength = textprogressbar(0,lastprintlength,i,n);

end %for

end %outer_iter
*/

int main(int argc, char *argv[])
{
	cudaSetDevice(0);
	
	int srsc = atoi(argv[1]);
	float lambda = static_cast<float>(atof(argv[2]));
	float gamma = static_cast<float>(atof(argv[3]));	
	int maxSingleIter = atoi(argv[4]);
	int maxIter = atoi(argv[5]);

	//bool verbose = false;
	//if (argc == 6)
	//	verbose = static_cast<bool>(atoi(argv[5]));
	const char *dataname = argv[6];

	//float lambda_l1 = 0.1f, lambda_l0 = 0.1f;
	//int maxSingleIter = 30, maxIter = 5;

	MATFile *srsc_input = matOpen((std::string("srsc_input_")+std::string(dataname)+".mat").c_str(),"r");
	mxArray *XArray = matGetVariable(srsc_input, "X");
	mxArray *ZArray = matGetVariable(srsc_input, "Z");
	mxArray *adjmatArray = matGetVariable(srsc_input, "adjmat");
	mxArray *adjweightArray = matGetVariable(srsc_input, "adjweight");
	mxArray *DArray = matGetVariable(srsc_input, "D");
	mxArray *DtDArray = matGetVariable(srsc_input, "DtD");
	mxArray *DtXArray = matGetVariable(srsc_input, "DtX");
	mxArray *LArray = matGetVariable(srsc_input, "L");
	//mxArray *knnArray = matGetVariable(srsc_input, "knn");

	float *h_X = static_cast<float*>(mxGetData(XArray));
	float *h_Z = static_cast<float*>( mxGetData(ZArray));
	float *h_adjmat = static_cast<float*>(mxGetData(adjmatArray));
	float *h_adjweight = static_cast<float*>(mxGetData(adjweightArray));
	float *h_D = static_cast<float*>( mxGetData(DArray));
	float *h_DtD = static_cast<float*>( mxGetData(DtDArray));
	float *h_DtX = static_cast<float*>( mxGetData(DtXArray));
	float L = *(static_cast<float*>( mxGetData(LArray)));
	//uint nn = (uint)(*(static_cast<float*>( mxGetData(knnArray))));

	const mwSize *Xsize = mxGetDimensions(XArray);
	uint d = static_cast<uint>(Xsize[0]);
	uint n = static_cast<uint>(Xsize[1]);
	const mwSize *Dsize = mxGetDimensions(DArray);
	uint nD = static_cast<uint>(Dsize[1]);
	const mwSize *adjmat_size_ = mxGetDimensions(adjmatArray);
	uint adjmat_size = static_cast<uint>(adjmat_size_[0]);

	//l0l1IO::process_input(h_X, h_l1graph_alpha, h_adjmat, h_A, h_AtA, S1, n, d, nn, "l0l1graph_input");
	
	float *d_X = NULL, *d_Z = NULL, *d_adjmat = NULL, *d_adjweight = NULL, *d_D = NULL, *d_DtD = NULL, *d_DtX = NULL;
	float *d_Zi_neighbor = NULL, *d_Wi_neighbor = NULL;
	float *d_Zi = NULL, *d_Zi0 = NULL, *d_Zi_proximal = NULL;
	float *d_L = NULL, *d_lambda = NULL, *d_lambda_L = NULL, *d_gamma = NULL;

	uint *d_neighbor_size = NULL;

	cudaMalloc((void**)&d_X,				sizeof(float)*d*n);
	cudaMalloc((void**)&d_Z,			sizeof(float)*nD*n);
	cudaMalloc((void**)&d_adjmat,			sizeof(float)*adjmat_size*n);
	cudaMalloc((void**)&d_adjweight,		sizeof(float)*adjmat_size*n);
	cudaMalloc((void**)&d_D,				sizeof(float)*d*nD);
	cudaMalloc((void**)&d_DtD,				sizeof(float)*nD*nD);
	cudaMalloc((void**)&d_DtX,				sizeof(float)*nD*n);
	cudaMalloc((void**)&d_Zi_neighbor,	sizeof(float)*nD*adjmat_size);
	cudaMalloc((void**)&d_Wi_neighbor,		sizeof(float)*adjmat_size);	
	cudaMalloc((void**)&d_Zi,			sizeof(float)*nD);
	cudaMalloc((void**)&d_Zi0,			sizeof(float)*nD);
	cudaMalloc((void**)&d_Zi_proximal,	sizeof(float)*nD);
	cudaMalloc((void**)&d_L,				sizeof(float));
	cudaMalloc((void**)&d_lambda,		sizeof(float));
	cudaMalloc((void**)&d_lambda_L,		sizeof(float));
	cudaMalloc((void**)&d_gamma,		sizeof(float));
	cudaMalloc((void**)&d_neighbor_size,	sizeof(uint)*n);

	cudaMemset(d_neighbor_size,0,sizeof(uint)*n);

	uint *h_neighbor_size = (uint*)malloc(sizeof(uint)*n);
	
	//debug
	//float *h_alphai = (float*)malloc(sizeof(float)*(nA));
	
	

	cudaMemcpy(d_X,h_X,sizeof(float)*d*n,cudaMemcpyHostToDevice);
	cudaMemcpy(d_Z,h_Z,sizeof(float)*nD*n,cudaMemcpyHostToDevice);
	cudaMemcpy(d_adjmat,h_adjmat,sizeof(float)*adjmat_size*n,cudaMemcpyHostToDevice);
	cudaMemcpy(d_adjweight,h_adjweight,sizeof(float)*adjmat_size*n,cudaMemcpyHostToDevice);
	cudaMemcpy(d_D,h_D,sizeof(float)*d*nD,cudaMemcpyHostToDevice);
	cudaMemcpy(d_DtD,h_DtD,sizeof(float)*nD*nD,cudaMemcpyHostToDevice);
	cudaMemcpy(d_DtX,h_DtX,sizeof(float)*nD*n,cudaMemcpyHostToDevice);
	
	float lambda_L = lambda/L;
	cudaMemcpy(d_L,&L,sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_lambda,&lambda,sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_lambda_L,&lambda_L,sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_gamma,&gamma,sizeof(float),cudaMemcpyHostToDevice);

	matClose(srsc_input);


	// create and initialize CUBLAS library object 
	cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);

	if (status != CUBLAS_STATUS_SUCCESS)
    {
        if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
            std::cout << "CUBLAS object instantialization error" << std::endl;
        }
        getchar ();
        return 0;
    }

	float elapsedTime = 0;
	
	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord(start, 0);	

	//compute AtX
	//compute_AtX(d_AtX, d_A, d_X, n, d, handle);

	compute_neighbor_size(d_neighbor_size, d_adjmat, adjmat_size, n);
	cudaMemcpy(h_neighbor_size,d_neighbor_size,sizeof(uint)*n,cudaMemcpyDeviceToHost);

	for (int outer_iter = 0; outer_iter < maxIter; outer_iter++)
	{
		for (uint i = 0; i < n; i++)
		{
			compute_Zi_neighbor(d_Zi_neighbor, d_Wi_neighbor, d_adjmat, d_adjweight, d_Z, i, nD, h_neighbor_size[i], adjmat_size);

			cudaMemcpy(d_Zi0,d_Z+i*nD,sizeof(float)*nD,cudaMemcpyDeviceToDevice);

			cudaMemcpy(d_Zi,d_Zi0,sizeof(float)*nD,cudaMemcpyDeviceToDevice);

			int iter = 0;
			while ( iter < maxSingleIter )
			{
				//debug				
				//export_variable(d_Zi_neighbor, "d_Zi_neighbor", nD, h_neighbor_size[i]);

				compute_Z_proximal(d_Zi_proximal, d_DtD, d_DtX, d_Zi, d_L, i, nD, handle);
								
				if (srsc)
					compute_sol_srsc(d_Zi, d_Zi_proximal, d_Zi_neighbor, d_Wi_neighbor, i, nD, h_neighbor_size[i], d_L, d_lambda, d_lambda_L, d_gamma);
				else
					compute_sol_l0_rsc(d_Zi, d_Zi_proximal, d_Zi_neighbor, d_Wi_neighbor, i, nD, h_neighbor_size[i], d_L, d_lambda, d_lambda_L, d_gamma);
				
				float err = 0.0f;
				compute_error_coef(&err, d_Zi, d_Zi0, nD, handle);

				//alphai0 = alphai;
				cudaMemcpy(d_Zi0,d_Zi,sizeof(float)*nD,cudaMemcpyDeviceToDevice);

				/*float obj = 0.0f, l2err = 0.0f, l1_spar_err = 0.0f, l0_spar_err = 0.0f;
				compute_obj_robust(obj, l2err, l1_spar_err, l0_spar_err, d_X, i, d_A, d_alphai, d_alphai_neighbor, d_W0_neighbor, lambda_l1, d_lambda_l0, n, nA, nn, d, handle);
				
				//debug
				//cudaMemcpy(h_obj,d_obj,sizeof(float),cudaMemcpyDeviceToHost);
				//cudaMemcpy(h_l2err,d_l2err,sizeof(float),cudaMemcpyDeviceToHost);
				//cudaMemcpy(h_l1_spar_err,d_l1_spar_err,sizeof(float),cudaMemcpyDeviceToHost);
				//cudaMemcpy(h_l0_spar_err,d_l0_spar_err,sizeof(float),cudaMemcpyDeviceToHost);

				if (verbose)
				{
					printf("proximal_manifold: errors = %.5f, iter: %d \n", err, iter);
					printf("obj is %.5f, l2err is %.5f, l1_spar_err is %.5f l0_spar_err is %.5f\n", obj, l2err, l1_spar_err, l0_spar_err);
				}*/

				iter = iter+1;

			}//while

			//alpha(:,i) = alphai;
			cudaMemcpy(d_Z+i*nD, d_Zi,sizeof(float)*nD,cudaMemcpyDeviceToDevice);

		}//for i = 1:n
		printf("iteration %d finished \n", outer_iter);
	}//for outer_iter
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
	//cudaDeviceSynchronize();
	//finishTime=clock();
	//elapsedTime =(float)(finishTime - startTime);

	// Clean up:
	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);
	printf("time to compute on gpu is %.10f second\n",elapsedTime/(CLOCKS_PER_SEC));




	MATFile *rFile = matOpen((std::string("srsc_result_")+std::string(dataname)+".mat").c_str(),"w");	
	
	mxArray* ZoutArray = mxCreateNumericMatrix(nD,n, mxSINGLE_CLASS, mxREAL);
	
	float *h_Zout   = (float*)mxGetData(ZoutArray);
	
	//transfer the data from gpu to cpu
	cudaMemcpy(h_Zout,d_Z,sizeof(float)*nD*n,cudaMemcpyDeviceToHost);

	matPutVariable(rFile, "Z", ZoutArray);
	matClose(rFile);

	mxDestroyArray(ZoutArray);

	//destroy the input matlab Arrays
	mxDestroyArray(XArray);
	mxDestroyArray(ZArray);
	mxDestroyArray(adjmatArray);
	mxDestroyArray(adjweightArray);
	mxDestroyArray(DArray);
	mxDestroyArray(DtDArray);
	mxDestroyArray(DtXArray);
	mxDestroyArray(LArray);
	//mxDestroyArray(knnArray);


	//deallocation

	free(h_neighbor_size);
	

	cudaFree(d_X);
	cudaFree(d_Z);
	cudaFree(d_adjmat);
	cudaFree(d_adjweight);
	cudaFree(d_D);
	cudaFree(d_DtD);
	cudaFree(d_DtX);
	cudaFree(d_Zi_neighbor);
	cudaFree(d_Wi_neighbor);	
	cudaFree(d_Zi);
	cudaFree(d_Zi0);
	cudaFree(d_Zi_proximal);
	cudaFree(d_L);
	cudaFree(d_lambda);
	cudaFree(d_lambda_L);
	cudaFree(d_gamma);
	cudaFree(d_neighbor_size);

	
	return 0;
}