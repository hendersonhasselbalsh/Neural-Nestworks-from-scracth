#pragma once

#include "../basic-includes.h"
#include "../01-interfaces/ILayer.h"
#include "../01-interfaces/IOptimizationLayer.h"



class Dropout : public ILayer, public IOptimizer {
	private:
		double _prob;
		Eigen::MatrixXd _dropOutVector;

	public:
		Dropout(double prob);

		Eigen::MatrixXd DropoutVector(size_t vectorSize);

		// Inherited via ILayer
		virtual Eigen::MatrixXd Forward(Eigen::MatrixXd& input) override; 
		virtual Eigen::MatrixXd Backward(Eigen::MatrixXd& dL_dO) override;
};
