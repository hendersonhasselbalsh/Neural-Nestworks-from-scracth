#pragma once

#include "ConvolutionCell.h"
#include "../mlp/multy-layer-perceptron.h"
#include "../utils/basic-includes.h"


class CNN {

	private:
		std::vector<IProcessingUnit*> _processingUnits;
		MLP _mlp;

		CNN();

	public:
		
		std::vector<double> Forward(Eigen::MatrixXd& input);


};

