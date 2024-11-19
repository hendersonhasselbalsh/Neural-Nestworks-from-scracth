#pragma once

#include "../basic-includes.h"
#include "../02-Dense-Layer/DenseLayer.h" 
#include "../03-Activation-Function/ActivationFunction.h"
#include "../05-Loss-Function/LossFunctions.h"
#include "../06-Managers/DataManager.h"
#include "../06-Managers/LayerManager.h"

class MLPbuilder;



class MLP {
	private:
		std::vector<ILayer*> _layers;
		ILossFunction* _lossFunc;
		size_t _max_epochs;
		long _batchSize;
		size_t _outputClasses;

		MLP();
	
	public:
		Eigen::MatrixXd CalculateOutput(Eigen::MatrixXd& inputs);
		Eigen::MatrixXd Backpropgagation(Eigen::MatrixXd& predictedY, Eigen::MatrixXd& correctY);
		Eigen::MatrixXd Backpropgagation(Eigen::MatrixXd& dL_dY);

		void Training(std::vector<std::pair<Eigen::MatrixXd, size_t>>& datas, std::function<void(void)> callback = [](){ });

		friend class MLPbuilder;
};

#include "../07-Builder/MLPbuilder.h"
