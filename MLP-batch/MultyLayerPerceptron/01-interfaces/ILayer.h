#pragma once

#include "../basic-includes.h"

#define __In__
#define __Out__


class ILayer {
	public:
		virtual Eigen::MatrixXd Forward(Eigen::MatrixXd& input) = 0;       // calculate output
		virtual Eigen::MatrixXd Backward(Eigen::MatrixXd& dL_dO) = 0;	  // backpropagation

		virtual void Initialize(__In__ size_t inputVecSize, __Out__ size_t* outputSize = nullptr) { (*outputSize) = inputVecSize; }
		virtual Eigen::MatrixXd AdamBackprop(Eigen::MatrixXd& dL_dO, double beta) { return Eigen::MatrixXd(); }
};
