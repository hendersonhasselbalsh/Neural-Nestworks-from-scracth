#pragma once

#include "../basic-includes.h"
#include "../01-interfaces/ILossFunction.h"
#include "../06-Managers/DataManager.h"


//----------------------------
//  MEAN SQUARE ERROR
//----------------------------

class MSE : public ILossFunction {
	public:
		// Inherited via ILossFunction
		virtual double f(Eigen::MatrixXd& predictedY, Eigen::MatrixXd& correctY) override;
		virtual Eigen::MatrixXd df(Eigen::MatrixXd& predictedY, Eigen::MatrixXd& correctY) override;

		virtual double Loss(Eigen::MatrixXd& predictedY, Eigen::MatrixXd& correctY) override;
		virtual Eigen::MatrixXd dLoss_dpredictedY(Eigen::MatrixXd& predictedY, Eigen::MatrixXd& correctY) override;

};
