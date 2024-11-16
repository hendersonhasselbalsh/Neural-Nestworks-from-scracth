#pragma once

#include "../basic-includes.h"


class ILayer {
	public:
		virtual Eigen::MatrixXd Forward(Eigen::MatrixXd input) = 0;       // calculate output
		virtual Eigen::MatrixXd Backward(Eigen::MatrixXd dL_dO) = 0;	  // backpropagation
};
