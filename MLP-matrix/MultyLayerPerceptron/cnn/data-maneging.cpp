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
