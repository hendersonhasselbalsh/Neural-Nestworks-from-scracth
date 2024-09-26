#pragma once

#include "ConvolutionCell.h"
#include "ActivationCell.h"
#include "pooling.h"
#include "data-maneging.h"
#include "layer-normalization.h"
#include "../mlp/multy-layer-perceptron.h"
#include "../utils/basic-includes.h"


class CNNbuilder;



class CNN {

	public:
		std::vector<IProcessingUnit*> _processingUnits;
		MLP _mlp;

		std::function<void(size_t, double, double&)> _UpdateLeraningRate;                      //  double f(size_t epoch, double accuracy, double currentLearningRate)

		size_t _reshapeRows;
		size_t _reshapeCols;

		size_t _maxEpochs;

		CNN();

	public:
		
		std::vector<double> Forward(Eigen::MatrixXd& input);
		std::vector<double> Backward(std::vector<double>& predictedValues, std::vector<double>& correctValues);

		void Training(std::vector<CNN_DATA> trainingDataSet, std::function<void(void)> callback = [](){ });

		void UpdateLearningRate(size_t epoch, double error);


	friend class CNNbuilder;
};




#include "CNNbuilder.h"
