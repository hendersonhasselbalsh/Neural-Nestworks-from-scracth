#pragma once

#include "../basic-includes.h"

class ILossFunction {
	public:
		// for individual vectors
		virtual double f(Eigen::MatrixXd& predictedY, Eigen::MatrixXd& correctY) = 0;
		virtual Eigen::MatrixXd df(Eigen::MatrixXd& predictedY, Eigen::MatrixXd& correctY) = 0;

		// for all the batch
		virtual double Loss(Eigen::MatrixXd& predictedY, Eigen::MatrixXd& correctY) = 0;
		virtual Eigen::MatrixXd dLoss_dpredictedY(Eigen::MatrixXd& predictedY, Eigen::MatrixXd& correctY) = 0;
};
