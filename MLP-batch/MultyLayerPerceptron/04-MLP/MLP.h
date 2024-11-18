#pragma once

#include "../basic-includes.h"
#include "../02-Dense-Layer/DenseLayer.h" 
#include "../02-Dense-Layer/DataManager.h"
#include "../03-Activation-Function/ActivationFunction.h"
#include "../05-Loss-Function/LossFunctions.h"


class MLP {
	public:
		std::vector<ILayer*> _layers;
		ILossFunction* _lossFunc;
		size_t max_epochs;
		size_t batchSize;

		MLP();
		static bool IsDenseLayer(ILayer* layer);
	
	public:
		Eigen::MatrixXd CalculateOutput(Eigen::MatrixXd& inputs);
		Eigen::MatrixXd Backpropgagation(Eigen::MatrixXd& predictedY, Eigen::MatrixXd& correctY);
		Eigen::MatrixXd Backpropgagation(Eigen::MatrixXd& dL_dY);

		void Training(std::vector<std::pair<Eigen::MatrixXd, size_t>> data, std::function<void(void)> callback = [](){ });
};

