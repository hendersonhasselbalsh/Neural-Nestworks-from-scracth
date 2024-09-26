#pragma once

#include "../utils/basic-includes.h"
#include "../utils/utils.h"
#include "../mlp/activation-functions.h"
#include "ProcessingUnity.h"



class ActivationCell : public IProcessingUnit {

	private:
		Eigen::Tensor<double, 3> _receivedInput;
		IActivationFunction* _actFunc;

	public:
		ActivationCell(IActivationFunction* actFunc);
		~ActivationCell();

		// Inherited via IProcessingUnit
		Eigen::Tensor<double, 3> Forward(Eigen::Tensor<double, 3>& input) override;
		Eigen::Tensor<double, 3> Backward(Eigen::Tensor<double, 3>& dLoss_dOutput) override;

};

