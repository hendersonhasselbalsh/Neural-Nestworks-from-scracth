#pragma once

#include "../basic-includes.h"
#include "../01-interfaces/ILayer.h"
#include "../01-interfaces/IOptimizationLayer.h"



class LayerNorm : public ILayer, public IOptimizer {
	private:
		double _learnRate;
		Eigen::MatrixXd _receivedInputBatch; 
		Eigen::MatrixXd _stdDevVec;
		Eigen::MatrixXd _meanVec;
		Eigen::MatrixXd _gamma;
		Eigen::MatrixXd _beta;

	public:
		LayerNorm(double learnRate = 0.01);

		Eigen::MatrixXd dNorm_dInput(Eigen::MatrixXd& inputBatch);
		Eigen::MatrixXd dLoss_dGamma(Eigen::MatrixXd& dLdN_batch);
		Eigen::MatrixXd dLoss_dBeta(Eigen::MatrixXd& dLdN_batch);

		// Inherited via ILayer
		virtual void Initialize(__In__ size_t inputVecSize, __Out__ size_t* outputSize = nullptr) override;
		virtual Eigen::MatrixXd Forward(Eigen::MatrixXd& inputBatch) override;
		virtual Eigen::MatrixXd Backward(Eigen::MatrixXd& dLdN_batch) override;
};
