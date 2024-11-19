#pragma once

class IActivationFunction {
	public:
		virtual double f(double x) { return x; }
		virtual double df(double x) { return x; }

		virtual Eigen::MatrixXd Activation(Eigen::MatrixXd& weitedSumVec) = 0;
		virtual Eigen::MatrixXd dActivation_dWeightedSum(Eigen::MatrixXd& weitedSumVec) = 0;
};
