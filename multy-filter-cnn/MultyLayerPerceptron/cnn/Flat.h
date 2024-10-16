#pragma once


#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../utils/basic-includes.h"
#include "../utils/utils.h"
#include "ProcessingUnity.h"




class FlatMatrix {

	private:
		size_t _matQnt;
		size_t _rows;
		size_t _cols;

	public:
		FlatMatrix();

		// Inherited via IProcessingUnit
		std::vector<double> Flat(Eigen::Tensor<double, 3>& input);
		Eigen::Tensor<double, 3> Reshape(std::vector<double>& dLoss_dOutput);

};




