#include "BatchNorm.h"


LayerNorm::LayerNorm(double learnRate)
{
	_learnRate = learnRate;
}


Eigen::MatrixXd LayerNorm::dNorm_dInput(Eigen::MatrixXd& inputBatch)
{
	double H = inputBatch.cols();
	Eigen::MatrixXd dN_dX = Eigen::MatrixXd(inputBatch.rows(), inputBatch.cols());

	for (size_t i = 0; i < inputBatch.rows(); i++) {

		double gamma = _gamma(i,0);
		double beta = _beta(i,0);
		double mean = _meanVec(i,0);
		double stdDev = _stdDevVec(i,0);

		Eigen::MatrixXd niiVec = (inputBatch.row(i).array() - mean);

		for (size_t j = 0; j < inputBatch.cols(); j++) {
			double nii = niiVec(0,j);

			//double val = ((gamma/stdDev) + (2.0*nii*nii*gamma / 2.0*stdDev*stdDev*stdDev*H)) * (1.0 - (1.0/H));
			double val = ((1.0/stdDev) - (2.0*nii*nii / 2.0*stdDev*stdDev*stdDev*H)) * (1.0 - (1.0/H));
			dN_dX(i,j) = 0.00001*val; 
		}
	}

	return dN_dX; 
}


Eigen::MatrixXd LayerNorm::dLoss_dGamma(Eigen::MatrixXd& dLdN_batch)
{
	Eigen::MatrixXd dLdGamma = Eigen::MatrixXd(_gamma.rows(), 1);

	for (size_t i = 0; i < dLdN_batch.rows(); i++) {
		double mean = _meanVec(i,0);
		double stdDev = _stdDevVec(i,0);
		Eigen::MatrixXd niiVec = (dLdN_batch.row(i).array() - mean);

		double sum = 0.0;
		for (size_t j = 0; j < dLdN_batch.cols(); j++) {
			double nii = niiVec(0,j);

			sum += dLdN_batch(i,j) * (nii/stdDev);
		}

		dLdGamma(i,0) = sum; 
	}
	
	return dLdGamma;
}


Eigen::MatrixXd LayerNorm::dLoss_dBeta(Eigen::MatrixXd& dLdN_batch)
{
	Eigen::MatrixXd dLdBeta = Eigen::MatrixXd(_beta.rows(), 1); 

	for (size_t i = 0; i < dLdN_batch.rows(); i++) {

		double sum = 0.0;
		for (size_t j = 0; j < dLdN_batch.cols(); j++) {

			sum += dLdN_batch(i, j);
		}

		dLdBeta(i, 0) = sum;
	}

	return dLdBeta;
}


void LayerNorm::Initialize(__In__ size_t inputVecSize, __Out__ size_t* outputSize)
{
	(*outputSize) = inputVecSize;

	_stdDevVec = Eigen::MatrixXd(inputVecSize, 1);
	_meanVec   = Eigen::MatrixXd(inputVecSize, 1); 
	_gamma     = Eigen::MatrixXd::Constant(inputVecSize, 1, 0.09); 
	_beta      = Eigen::MatrixXd::Constant(inputVecSize, 1, 0.0);
}


Eigen::MatrixXd LayerNorm::Forward(Eigen::MatrixXd& inputBatch)
{
	_receivedInputBatch = inputBatch;

	double EPSON = 0.0;
	Eigen::MatrixXd normalizedMatrix = Eigen::MatrixXd(inputBatch.rows(), inputBatch.cols());

    for (int i = 0; i < inputBatch.rows(); i++) {  
		double mean = inputBatch.row(i).mean();
		_meanVec(i, 0) = mean;

		double variance = (inputBatch.row(i).array() - mean).square().sum() / inputBatch.cols(); 
        double stdDev = std::sqrt(variance);
		_stdDevVec(i,0) = stdDev;

		Eigen::MatrixXd norm = (inputBatch.row(i).array() - mean) / (stdDev + EPSON);
		Eigen::MatrixXd scaled = norm * _gamma(i, 0);
		normalizedMatrix.row(i) = (scaled.array() + _beta(i,0)); 
		//normalizedMatrix.row(i) = scaled;
    }

	return normalizedMatrix;
}


Eigen::MatrixXd LayerNorm::Backward(Eigen::MatrixXd& dLdN_batch)
{
	Eigen::MatrixXd dN_dX = dNorm_dInput(_receivedInputBatch);
	Eigen::MatrixXd dL_dX = dLdN_batch.cwiseProduct(dN_dX);

	Eigen::MatrixXd dL_dGamma = dLoss_dGamma(dLdN_batch); 
	_gamma = _gamma - _learnRate * dL_dGamma; 

	Eigen::MatrixXd dL_dBeta = dLoss_dBeta(dLdN_batch);
	_beta = _beta - _learnRate * dL_dBeta;  

	for (size_t i = 0; i < dL_dX.rows(); i++) {
		for (size_t j = 0; j < dL_dX.cols(); j++) {
			if (std::abs(dL_dX(i,j)) >= 2.0) { std::cout << ">>> BUG dL_dX too big\n"; break; }
		}
	}

	return dL_dX;
}
