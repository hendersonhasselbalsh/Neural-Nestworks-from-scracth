#pragma once

#include "../basic-includes.h"
#include "../01-interfaces/ILayer.h"
#include "../01-interfaces/IActivationFunction.h"
#include "../06-Managers/DataManager.h"



class Softmax : public ILayer, public IActivationFunction {
	private:
		Eigen::MatrixXd _activatedBatch;

	public:
		Softmax();

		// Inherited via IActivationFunction
		virtual Eigen::MatrixXd Activation(Eigen::MatrixXd& weitedSumVec) override;
		virtual Eigen::MatrixXd dActivation_dWeightedSum(Eigen::MatrixXd& weitedSumVec) override;

		// Inherited via ILayer
		virtual Eigen::MatrixXd Forward(Eigen::MatrixXd& weightedSumBatch) override;
		virtual Eigen::MatrixXd Backward(Eigen::MatrixXd& dLdA_batch) override;

};
