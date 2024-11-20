#pragma once

#include "../basic-includes.h"
#include "../01-interfaces/ILayer.h"
#include "../01-interfaces/IActivationFunction.h"
#include "../06-Managers/DataManager.h"


class Tanh : public ILayer, public IActivationFunction {
	private:
		Eigen::MatrixXd _receivedInputBatch;

	public:
		Tanh();

		// Inherited via IActivationFunction
		virtual double f(double x) override;
		virtual double df(double x) override;
		virtual Eigen::MatrixXd Activation(Eigen::MatrixXd& weitedSumVec) override;
		virtual Eigen::MatrixXd dActivation_dWeightedSum(Eigen::MatrixXd& weitedSumVec) override;

		// Inherited via ILayer
		virtual Eigen::MatrixXd Forward(Eigen::MatrixXd& weightedSumBatch) override;
		virtual Eigen::MatrixXd Backward(Eigen::MatrixXd& dLdA_batch) override;


};

