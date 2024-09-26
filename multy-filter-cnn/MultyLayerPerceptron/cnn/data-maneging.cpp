#include "data-maneging.h"


//----------------------
//   SCALE
//----------------------

/*
Scale::Scale(double rangeStart, double rangeEnd)
	: _rangeStart(rangeStart), _rangeEnd(rangeEnd)
{
}

Scale::~Scale()
{
}

Eigen::MatrixXd Scale::Forward(Eigen::MatrixXd& input)
{
	_max = input.maxCoeff();
	_min = input.minCoeff();

	for (size_t i = 0; i < input.rows(); i++) {
		for (size_t j = 0; j < input.cols(); j++) {
			input(i,j)  =  (_rangeEnd - _rangeStart) * ( (input(i,j) - _min)/(_max - _min) ) + _rangeStart;
		}
	}

	return (Eigen::MatrixXd) input;
}

Eigen::MatrixXd Scale::Backward(Eigen::MatrixXd& dLoss_dOutput)
{
	for (size_t i = 0; i < dLoss_dOutput.rows(); i++) {
		for (size_t j = 0; j < dLoss_dOutput.cols(); j++) {
			dLoss_dOutput(i, j)  =  ((_rangeEnd - _rangeStart)/(_max - _min)) * dLoss_dOutput(i, j);
		}
	}

	return (Eigen::MatrixXd) dLoss_dOutput;
}
*/







//----------------------
//   Normalize
//----------------------

Normalize::Normalize()
{
}

Normalize::~Normalize()
{
}

Eigen::MatrixXd Normalize::Forward(Eigen::MatrixXd& input)
{
	_mean = Normalize::Mean(input);
	_std_dev = Normalize::StandartDeviation(input);

	for (size_t i = 0; i < input.rows(); i++) {
		for (size_t j = 0; j < input.cols(); j++) {
			input(i,j)  =  (input(i,j) - _mean) / _std_dev;
		}
	}

	return input;
}

Eigen::MatrixXd Normalize::Backward(Eigen::MatrixXd& dLoss_dOutput)
{
	for (size_t i = 0; i < dLoss_dOutput.rows(); i++) {
		for (size_t j = 0; j < dLoss_dOutput.cols(); j++) {
			dLoss_dOutput(i, j)  =  dLoss_dOutput(i, j) * (1.0/ _std_dev);
		}
	}

	return dLoss_dOutput;
}



double Normalize::Mean(Eigen::MatrixXd& matrix)
{
	return matrix.mean();
}

double Normalize::StandartDeviation(Eigen::MatrixXd& matrix)
{
	double m = matrix.mean();

	Eigen::MatrixXd diff = matrix.array() - m;
	double variance = (diff.array().square().sum()) / (matrix.size() - 1);

	double epson  =  1e-10;

	return std::sqrt(variance + epson);
}




//--------------
// NORMALIZATION
//--------------

Normalization::Normalization(double learningRate)
{
	 _learningRate = learningRate;
	 _betas = 0.00001;
	 _gammas = 0.9;
}

Normalization::~Normalization()
{
}

Eigen::MatrixXd Normalization::LayerNorm(Eigen::MatrixXd& input)
{
	double epson = 1e-5;

	Eigen::MatrixXd normalized = input;

	_layerMeans = input.mean();
	double variance = (input.array() - _layerMeans).square().mean();
	_layerStddev = std::sqrt(variance + epson);

	normalized = (input.array() - _layerMeans) / _layerStddev;
	normalized = (normalized.array() * _gammas) + _betas;

	return normalized;
}



Eigen::MatrixXd Normalization::Forward(Eigen::MatrixXd& input)
{
	_layerMeans = 0.0;
	_layerStddev = 0.0;

	_receivecInput  =  input;

	Eigen::MatrixXd normalized = LayerNorm(_receivecInput);

	return normalized;
}



Eigen::MatrixXd Normalization::Backward(Eigen::MatrixXd& dL_dNormalized)
{
	// --- dL_dBetta and dL_dGamma
	double dL_dBeta = 0.0;
	double dL_dGamma = 0.0;

	for (size_t row = 0; row < dL_dNormalized.rows(); row++) {
		for (size_t col = 0; col < dL_dNormalized.cols(); col++) {
			// --- dL_dBetta
			dL_dBeta  +=  dL_dNormalized(row, col) * 1.0;

			// --- dL_dGama
			double x  =  _receivecInput(row, col);
			double nii = x - _layerMeans;
			dL_dGamma +=  dL_dNormalized(row, col) * (nii / _layerStddev);
		}
	}

	_betas  =  _betas  -  _learningRate * dL_dBeta;
	_gammas  =  _gammas  -  _learningRate * dL_dGamma;


	//--- dL_dX
	double dL_dVariance  =  DL_DVariance(dL_dNormalized);
	Eigen::MatrixXd dL_dNii  =  DL_DNii(dL_dVariance, dL_dNormalized);
	double dL_dMeans  =  DL_DMeans(dL_dNormalized);
	Eigen::MatrixXd dL_dInputMatrix  =  DL_DInput(dL_dMeans, dL_dNii);


	return dL_dInputMatrix;
}



double Normalization::DL_DVariance(Eigen::MatrixXd& dL_dNormalized)
{
	double dL_dVariance = 0.0;

	for (size_t row = 0; row < dL_dNormalized.rows(); row++) { 
		for (size_t col = 0; col < dL_dNormalized.cols(); col++) {
			double x = _receivecInput(row, col);
			double nii_element  =  x - _layerMeans;

			dL_dVariance  +=  dL_dNormalized(row, col) * (-nii_element / 2.0*std::pow(_layerStddev, 3));
		}
	}

	return dL_dVariance;
}


Eigen::MatrixXd Normalization::DL_DNii(double& dL_dVariance, Eigen::MatrixXd& dL_dNormalized)
{
	assert(dL_dVariance.rows()==1 && "[ERROR]: dL_dVariance must be a horizontal vector");
	assert(dL_dVariance.cols()==_receivecInput.cols() && "[ERROR]: each col must have their dL_dVariance");


	Eigen::MatrixXd dL_dNii  =  Eigen::MatrixXd(_receivecInput.rows(), _receivecInput.cols());

	double rowSize  =  (double)_receivecInput.rows();

	for (size_t row = 0; row < _receivecInput.rows(); row++) {
		for (size_t col = 0; col < _receivecInput.cols(); col++) {
			double x  =  _receivecInput(row, col);
			double nii  =  x  -  _layerMeans;

			dL_dNii(row, col) = (2*nii / rowSize) * dL_dVariance   +   dL_dNormalized(row, col) * (_gammas/_layerStddev);
		}
	}

	return dL_dNii;
}


double Normalization::DL_DMeans(Eigen::MatrixXd& dL_dNormalized)
{
	double dL_dMeans = 0.0;

	for (size_t row = 0; row < _receivecInput.rows(); row++) {
		for (size_t col = 0; col < _receivecInput.cols(); col++) {
			dL_dMeans  +=  -dL_dNormalized(row, col) * (_gammas / _layerStddev);
		}
	}

	return dL_dMeans;
}


Eigen::MatrixXd Normalization::DL_DInput(double& dL_dMean, Eigen::MatrixXd& dL_dNii)
{
	Eigen::MatrixXd dL_dInput = Eigen::MatrixXd(_receivecInput.rows(), _receivecInput.cols()); 

	double rowSize  =  (double)_receivecInput.rows();

	for (size_t row = 0; row < _receivecInput.rows(); row++) {
		for (size_t col = 0; col < _receivecInput.cols(); col++) {
			dL_dInput(row, col)  =  dL_dMean * (1.0/rowSize)   +   dL_dNii(row, col) * (1.0 - (1.0/rowSize));
		}
	}

	return dL_dInput;
}



