#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "../utils/basic-includes.h"
#include "../utils/utils.h"
#include "../mlp/activation-functions.h"


class IProcessingUnit {
	public: 
		virtual Eigen::Tensor<double, 3> Forward(Eigen::Tensor<double, 3>& input) = 0;
		virtual Eigen::Tensor<double, 3> Backward(Eigen::Tensor<double, 3>& dLoss_dOutput) = 0;
		virtual void UpdateLearningRate(size_t epoch, double error, std::function<void(size_t, double, double&)> UpdateRule) { }

		//virtual Eigen::MatrixXd Forward(Eigen::MatrixXd& input) = 0;            /*pure virtual method*/
		//virtual Eigen::MatrixXd Backward(Eigen::MatrixXd& dLoss_dOutput) = 0;   /*pure virtual method*/
};


