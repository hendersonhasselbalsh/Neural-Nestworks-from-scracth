#pragma once

#include "../utils/basic-includes.h"
#include "../utils/utils.h"
#include "../mlp/activation-functions.h"
#include "ProcessingUnity.h"



class ActivationCell : public IProcessingUnit {

	private:
		Eigen::MatrixXd _receivedInput;
		IActivationFunction* _actFunc;

	public:
		ActivationCell(IActivationFunction* actFunc);
		~ActivationCell();

		// Inherited via IProcessingUnit
		Eigen::MatrixXd Forward(Eigen::MatrixXd& input) override;
		Eigen::MatrixXd Backward(Eigen::MatrixXd& dLoss_dOutput) override;

};

