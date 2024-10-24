#pragma once

#include "../utils/basic-includes.h"
#include "../utils/utils.h"
#include "ProcessingUnity.h"


//--------------
// MAX POOLING
//--------------

class Scale : public IProcessingUnit {

	private:
		double _rangeStart;
		double _rangeEnd;

		double _max;
		double _min;

	public:
		//ConvolutionCell(size_t poolSize, double learnRate = 0.01);
		Scale(double rangeStart, double rangeEnd);
		~Scale();


		// Inherited via IProcessingUnit
		Eigen::MatrixXd Forward(Eigen::MatrixXd& input) override;
		Eigen::MatrixXd Backward(Eigen::MatrixXd& dLoss_dOutput) override;

};



class Normalize : public IProcessingUnit {

	private:
		double _mean;
		double _std_dev;

	public:
		//ConvolutionCell(size_t poolSize, double learnRate = 0.01);
		Normalize();
		~Normalize();


		// Inherited via IProcessingUnit
		Eigen::MatrixXd Forward(Eigen::MatrixXd& input) override;
		Eigen::MatrixXd Backward(Eigen::MatrixXd& dLoss_dOutput) override;


		static double Mean(Eigen::MatrixXd& matrix);
		static double StandartDeviation(Eigen::MatrixXd& matrix);

};


