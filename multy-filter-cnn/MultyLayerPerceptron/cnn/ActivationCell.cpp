#include "ActivationCell.h"



ActivationCell::ActivationCell(IActivationFunction* actFunc)
{
	_actFunc = actFunc;
}


ActivationCell::~ActivationCell()
{
}



Eigen::Tensor<double, 3> ActivationCell::Forward(Eigen::Tensor<double, 3>& input)
{
	_receivedInput  =  input;

	size_t deep = _receivedInput.dimension(0);
	size_t rows = _receivedInput.dimension(1);
	size_t cols = _receivedInput.dimension(2);

	Eigen::Tensor<double, 3> activatedMatrix  =  Eigen::Tensor<double, 3>(deep, rows, cols);

	for (size_t z = 0; z < deep; z++) {
		for (size_t i = 0; i < rows; i++) {
			for (size_t j = 0; j < cols; j++) {
				activatedMatrix(z, i, j)  =  _actFunc->f( activatedMatrix(z, i, j) );
			}
		}
	}

	return activatedMatrix;
}

//Eigen::MatrixXd ActivationCell::Forward(Eigen::MatrixXd& input)
//{
//	_receivedInput = input;
//
//	size_t rows = _receivedInput.rows();
//	size_t cols = _receivedInput.cols();
//
//	Eigen::MatrixXd activatedMatrix  =  Eigen::MatrixXd(rows, cols);
//
//	for (size_t i = 0; i < rows; i++) {
//		for (size_t j = 0; j < cols; j++) {
//			double activatedValue = _actFunc->f( _receivedInput(i,j) );
//			activatedMatrix(i,j)  =  activatedValue;
//		}
//	}
//
//	return activatedMatrix;
//}



Eigen::Tensor<double, 3> ActivationCell::Backward(Eigen::Tensor<double, 3>& dLoss_dOutput)
{
	size_t deep = _receivedInput.dimension(0);
	size_t rows = _receivedInput.dimension(1);
	size_t cols = _receivedInput.dimension(2);

	Eigen::Tensor<double, 3> dLoss_dActiv  =  Eigen::Tensor<double, 3>(deep, rows, cols);

	for (size_t z = 0; z < deep; z++) {
		for (size_t i = 0; i < rows; i++) {
			for (size_t j = 0; j < cols; j++) {
				double dOutput_dAct  =  _actFunc->df( _receivedInput(z, i, j) ); 
				dLoss_dActiv(z, i, j)  =  dLoss_dOutput(z, i, j) * dOutput_dAct;  
			}
		}
	}

	return dLoss_dActiv;
}


//Eigen::MatrixXd ActivationCell::Backward(Eigen::MatrixXd& dLoss_dOutput)
//{
//	assert(_receivedInput.size() == dLoss_dOutput.size()  &&  "those matrix Should have the same dimensions");
//
//	size_t rows = _receivedInput.rows();
//	size_t cols = _receivedInput.cols();
//
//	Eigen::MatrixXd dLoss_dActivated  =  Eigen::MatrixXd(rows, cols);
//
//	for (size_t i = 0; i < rows; i++) {
//		for (size_t j = 0; j < cols; j++) {
//			double activatedValue = _actFunc->df(_receivedInput(i, j)) * dLoss_dOutput(i,j);
//			dLoss_dActivated(i, j)  =  activatedValue;
//		}
//	}
//
//	return dLoss_dActivated;
//}
