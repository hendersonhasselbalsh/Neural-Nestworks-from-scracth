#include "Flat.h"

FlatMatrix::FlatMatrix()
{
	_matQnt = 0;
	_rows = 0;
	_cols = 0;
}



std::vector<double> FlatMatrix::Flat(Eigen::Tensor<double, 3>& input)
{
	_matQnt  =  input.dimension(0);
	_rows  =  input.dimension(1);
	_cols  =  input.dimension(2);

	std::vector<double> output = std::vector<double>();

	for (size_t ip = 0; ip < _matQnt; ip++) {
		auto mat  =  Utils::TensorSlice(input, ip);
		auto flatMat = Utils::FlatMatrix(mat);

		output.insert(output.end(), flatMat.begin(), flatMat.end()); 
	}

	return output;
}



Eigen::Tensor<double, 3> FlatMatrix::Reshape(std::vector<double>& dLoss_dOutput)
{
	size_t matSize = _rows * _cols;
	Eigen::Tensor<double,3> output = Eigen::Tensor<double,3>(_matQnt, _rows, _cols);

	for (size_t i = 0; i < _matQnt; i++) {
		size_t start = i*matSize;
		size_t end = (i+1)*matSize;
		std::vector<double> flated = std::vector<double>(dLoss_dOutput.begin() + start, dLoss_dOutput.begin() + end);

		Eigen::MatrixXd reshaped = Utils::ReshapeMatrix(flated, _rows, _cols);
		output.chip(i, 0) = reshaped;
	}

	return output;
}

