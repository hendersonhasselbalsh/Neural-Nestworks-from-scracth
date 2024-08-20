#pragma once

#include "ConvolutionCell.h"
#include "../mlp/multy-layer-perceptron.h"
#include "../utils/basic-includes.h"


class CNN {

	private:
		std::vector<IProcessingUnit*> _processingUnits;
		MLP _mlp;

		size_t _reshapeRows;
		size_t _reshapeCols;

		CNN();

	public:
		
		std::vector<double> Forward(Eigen::MatrixXd& input);
		std::vector<double> Backward(std::vector<double>& predictedValues, std::vector<double>& correctValues);


};

