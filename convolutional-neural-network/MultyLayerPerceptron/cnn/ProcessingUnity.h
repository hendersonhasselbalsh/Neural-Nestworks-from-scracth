#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include "../utils/basic-includes.h"
#include "../utils/utils.h"
#include "../mlp/activation-functions.h"


class IProcessingUnit {
	public: 
		virtual Eigen::MatrixXd Forward(Eigen::MatrixXd& input) = 0;            /*pure virtual method*/
		virtual Eigen::MatrixXd Backward(Eigen::MatrixXd& dLoss_dOutput) = 0;   /*pure virtual method*/
};
