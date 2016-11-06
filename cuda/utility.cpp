#include "utility.h"


namespace SrSCIO
{
	void process_input(float*& X, float*& l1graph_alpha, float*& adjmat, float*& A, float*& AtA, float& S1, uint& n, uint& d, uint &nn, const char *input_matfile_name)
	{
		MATFile *input = matOpen("l0l1graph_input.mat","r");
		mxArray *XArray = matGetVariable(input, "X");
		mxArray *l1graph_alphaArray = matGetVariable(input, "l1graph_alpha");
		mxArray *adjmatArray = matGetVariable(input, "adjmat");
		mxArray *AArray = matGetVariable(input, "A");
		mxArray *AtAArray = matGetVariable(input, "AtA");
		mxArray *S1Array = matGetVariable(input, "S1");
		mxArray *knnArray = matGetVariable(input, "knn");

		X = (float*) mxGetData(XArray);
		l1graph_alpha = (float*) mxGetData(l1graph_alphaArray);
		adjmat = (float*) mxGetData(adjmatArray);
		A = (float*) mxGetData(AArray);
		AtA = (float*) mxGetData(AtAArray);
		S1 = *((float*) mxGetData(S1Array));
		float *knn = (float*) mxGetData(knnArray);

		const mwSize *Xsize = mxGetDimensions(XArray);
		d = (uint)Xsize[0];
		n = (uint)Xsize[1];

		nn = (uint)(*knn);



		//mxDestroyArray(XArray);
		//mxDestroyArray(l1graph_alphaArray);
		//mxDestroyArray(W0Array);
		//mxDestroyArray(AArray);
		//mxDestroyArray(AtAArray);
		//mxDestroyArray(S1Array);
		//mxDestroyArray(knnArray);
	}
}

#ifdef _MY_DEBUG
void export_variable(const float *d_data, const char *name, uint h, uint w)
{
	MATFile *rFile = matOpen((std::string(name)+".mat").c_str(),"w");	
	
	mxArray* dataArray = mxCreateNumericMatrix(h,w, mxSINGLE_CLASS, mxREAL);
	
	float *h_data = (float*)mxGetData(dataArray);
	
	//transfer the data from gpu to cpu
	cudaMemcpy(h_data,d_data,sizeof(float)*h*w,cudaMemcpyDeviceToHost);

	matPutVariable(rFile, name, dataArray);
	matClose(rFile);
}
#endif