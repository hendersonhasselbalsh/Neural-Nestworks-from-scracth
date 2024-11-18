#pragma once

#include "../basic-includes.h"
#include "../01-interfaces/ILayer.h"
#include "../01-interfaces/IActivationFunction.h"



//------------------------------
//  LINEAR
//------------------------------

class Linear : public ILayer, public IActivationFunction {
	public:
		//Eigen::MatrixXd* _receivedBatchWeightedSum;	

	public:
		Linear();
		~Linear();

		// Inherited via ILayer
		virtual Eigen::MatrixXd Forward(Eigen::MatrixXd& batchWeightedSum) override;
		virtual Eigen::MatrixXd Backward(Eigen::MatrixXd& dL_dbatchActivation) override;

};



//------------------------------
//  Sigmoid
//------------------------------

class Sigmoid : public ILayer, public IActivationFunction {
	public:
		Eigen::MatrixXd _activatedBach;

	public:
		Sigmoid();
		~Sigmoid();

		static double f_sigmoid(double x);

		// Inherited via ILayer
		virtual Eigen::MatrixXd Forward(Eigen::MatrixXd& batchWeightedSum) override;
		virtual Eigen::MatrixXd Backward(Eigen::MatrixXd& dL_dbatchActivation) override;

};
