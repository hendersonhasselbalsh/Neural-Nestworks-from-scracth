#pragma once

#include "../utils/basic-includes.h"
#include "../utils/utils.h"
#include "ProcessingUnity.h"


class ConvolutionCell : public IProcessingUnit {
	
	private:
		double _learningRate;
		Eigen::MatrixXd _filter;
		Eigen::MatrixXd _receivedInput;

	public:
		//ConvolutionCell(size_t filterSize, double learnRate = 0.01);
		ConvolutionCell(size_t filterRow, size_t filterCol, double learnRate = 0.01);
		~ConvolutionCell();

		static Eigen::MatrixXd Convolute(Eigen::MatrixXd& input, Eigen::MatrixXd& filter);
		static Eigen::MatrixXd Convolute(Eigen::MatrixXd& input, Eigen::MatrixXd& filter, size_t padding);


		// Inherited via IProcessingUnit
		Eigen::MatrixXd Forward(Eigen::MatrixXd& input) override;
		Eigen::MatrixXd Backward(Eigen::MatrixXd& dLoss_dOutput) override;

};
