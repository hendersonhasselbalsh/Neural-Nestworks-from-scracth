#pragma once

#include "../basic-includes.h"
#include "../01-interfaces/ILayer.h"
#include "../01-interfaces/IActivationFunction.h"
#include "../06-Managers/DataManager.h"



//------------------------------
//  Sigmoid
//------------------------------

class Sigmoid : public ILayer, public IActivationFunction {
	public:
		Eigen::MatrixXd _activatedBach;

	public:
		Sigmoid();
		~Sigmoid();

		// Inherited via IActivationFunction
		virtual double f(double x) override;
		virtual Eigen::MatrixXd Activation(Eigen::MatrixXd& weitedSumVec) override;
		virtual Eigen::MatrixXd dActivation_dWeightedSum(Eigen::MatrixXd& weitedSumVec) override;

		// Inherited via ILayer
		virtual Eigen::MatrixXd Forward(Eigen::MatrixXd& batchWeightedSum) override;
		virtual Eigen::MatrixXd Backward(Eigen::MatrixXd& dL_dbatchActivation) override;
};
