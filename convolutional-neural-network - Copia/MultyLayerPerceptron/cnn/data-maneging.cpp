#include "data-maneging.h"


//----------------------
//   SCALE
//----------------------


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

	 _layerMeans = 0.0;
	 _layerStddev = 0.0; 

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
	//normalized = (normalized.array() * _gammas) + _betas;

	return normalized;
}






Eigen::MatrixXd Normalization::DLoss_DstdDev(Eigen::MatrixXd& dLoss_dy)
{
	double dy_dStdDev  =  - (1.0/std::pow(_layerStddev, 2));
	Eigen::MatrixXd dLoss_dStdDev  =  dLoss_dy *  dy_dStdDev;

	return dLoss_dStdDev;
}

Eigen::MatrixXd Normalization::DLoss_DVar(Eigen::MatrixXd& dLoss_dStdDev)
{
	double dStdDev_dVar  =  -1.0/(2.0 * _layerStddev);

	Eigen::MatrixXd dLoss_dVar  =  dLoss_dStdDev * dStdDev_dVar;

	return dLoss_dVar;
}


Eigen::MatrixXd Normalization::DLoss_DNii(Eigen::MatrixXd& dLoss_dVar)
{
	double size  =  (double)(dLoss_dVar.rows(), dLoss_dVar.cols());

	Eigen::MatrixXd dVar_dNii  =  Eigen::MatrixXd(dLoss_dVar.rows(), dLoss_dVar.cols());

	for (size_t i = 0; i < dLoss_dVar.rows(); i++) {
		for (size_t j = 0; j < dLoss_dVar.cols(); j++) {
			double x = _receivecInput(i,j);
			double nii = x - _layerMeans;

			dVar_dNii(i,j) = (2.0/size) * nii;
		}
	}

	Eigen::MatrixXd dLoss_dNii  =  dLoss_dVar.array() * dVar_dNii.array();

	return dLoss_dNii;
}

Eigen::MatrixXd Normalization::DLoss_Dx(Eigen::MatrixXd& dLoss_dNii)
{
	double size  =  (double)(dLoss_dNii.rows(), dLoss_dNii.cols());
	double dNii_dInput  =  1.0 - (1.0/size); 

	Eigen::MatrixXd dLoss_dInput  =  dLoss_dNii * dNii_dInput;

	return dLoss_dInput;
}


Eigen::MatrixXd Normalization::DLoss_DInput(Eigen::MatrixXd& dLoss_dy)
{
	// first route
	double size  =  (double)(dLoss_dy.rows(), dLoss_dy.cols()); 

	double dy_dNii = 1.0/_layerStddev;
	double dNii_dInput  =  1.0 - (1.0/size);
	Eigen::MatrixXd dLoss_dInput = dLoss_dy * dy_dNii * dNii_dInput;


	// second route
	auto dLoss_dStdDev = DLoss_DstdDev(dLoss_dy);
	auto dLoss_dVar  =  DLoss_DVar(dLoss_dStdDev);
	auto dLoss_dNii  =  DLoss_DNii(dLoss_dVar);
	auto dLoss_dx  =  DLoss_Dx(dLoss_dNii);

	dLoss_dInput = dLoss_dInput + dLoss_dx;



	return dLoss_dInput;
}




Eigen::MatrixXd Normalization::Forward(Eigen::MatrixXd& input)
{
	/*_layerMeans = 0.0;
	_layerStddev = 0.0;*/

	_receivecInput  =  input;

	Eigen::MatrixXd normalized = LayerNorm(_receivecInput);

	return normalized;
}



Eigen::MatrixXd Normalization::Backward(Eigen::MatrixXd& dL_dNormalized)
{
	// --- dL_dBetta and dL_dGamma
	//double dL_dBeta = 0.0;
	//double dL_dGamma = 0.0;

	//for (size_t row = 0; row < dL_dNormalized.rows(); row++) {
	//	for (size_t col = 0; col < dL_dNormalized.cols(); col++) {
	//		// --- dL_dBetta
	//		dL_dBeta  +=  dL_dNormalized(row, col) * 1.0;

	//		// --- dL_dGama
	//		double x  =  _receivecInput(row, col);
	//		double nii = x - _layerMeans;
	//		dL_dGamma +=  dL_dNormalized(row, col) * (nii / _layerStddev);
	//	}
	//}
	//
    //_betas  =  _betas  -  _learningRate * dL_dBeta;
	//_gammas  =  _gammas  -  _learningRate * dL_dGamma;


	////--- dL_dX

	auto dL_dInputMatrix  =  DLoss_DInput(dL_dNormalized);
	return dL_dInputMatrix;
}
