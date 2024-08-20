#pragma once

#include "ConvolutionCell.h"
#include "../mlp/multy-layer-perceptron.h"
#include "../utils/basic-includes.h"


class CNNbuilder;



class CNN {

	private:
		std::vector<IProcessingUnit*> _processingUnits;
		MLP _mlp;

		size_t _reshapeRows;
		size_t _reshapeCols;
		size_t _maxEpochs;

		CNN();

	public:
		
		std::vector<double> Forward(Eigen::MatrixXd& input);
		std::vector<double> Backward(std::vector<double>& predictedValues, std::vector<double>& correctValues);

	friend class CNNbuilder;
};




#include "CNNbuilder.h"
