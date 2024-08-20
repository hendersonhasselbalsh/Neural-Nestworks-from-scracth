#include "ActivationCell.h"



ActivationCell::ActivationCell(IActivationFunction* actFunc)
{
	_actFunc = actFunc;
}


ActivationCell::~ActivationCell()
{
}

Eigen::MatrixXd ActivationCell::Forward(Eigen::MatrixXd& input)
{
	_receivedInput = input;

	size_t rows = _receivedInput.rows();
	size_t cols = _receivedInput.cols();

	Eigen::MatrixXd activatedMatrix  =  Eigen::MatrixXd(rows, cols);

	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			double activatedValue = _actFunc->f( _receivedInput(i,j) );
			activatedMatrix(i,j)  =  activatedValue;
		}
	}

	return activatedMatrix;
}


Eigen::MatrixXd ActivationCell::Backward(Eigen::MatrixXd& dLoss_dOutput)
{
	assert(_receivedInput.size() != dLoss_dOutput.size()  &&  "those matrix Should have the same dimensions");

	size_t rows = _receivedInput.rows();
	size_t cols = _receivedInput.cols();

	Eigen::MatrixXd dLoss_dActivated  =  Eigen::MatrixXd(rows, cols);

	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			double activatedValue = _actFunc->df(_receivedInput(i, j)) * dLoss_dOutput(i,j);
			dLoss_dActivated(i, j)  =  activatedValue;
		}
	}

	return dLoss_dActivated;
}
