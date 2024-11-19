#pragma once

#include "../basic-includes.h"
#include "../01-interfaces/ILayer.h"
#include "../01-interfaces/IActivationFunction.h"
#include "../06-Managers/DataManager.h"



//------------------------------
//  ReLU
//------------------------------

class ReLU : public ILayer, public IActivationFunction {
	public:
		Eigen::MatrixXd _receivedBachInput;

	public:
		ReLU();

		// Inherited via IActivationFunction
		virtual double f(double x) override;
		virtual double df(double x) override;
		virtual Eigen::MatrixXd Activation(Eigen::MatrixXd& weitedSumVec) override;
		virtual Eigen::MatrixXd dActivation_dWeightedSum(Eigen::MatrixXd& weitedSumVec) override;

		// Inherited via ILayer
		virtual Eigen::MatrixXd Forward(Eigen::MatrixXd& weightedSumBatch) override;
		virtual Eigen::MatrixXd Backward(Eigen::MatrixXd& dLdA_batch) override;

};
