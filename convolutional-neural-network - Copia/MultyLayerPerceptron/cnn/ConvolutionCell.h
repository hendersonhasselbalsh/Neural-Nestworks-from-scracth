#pragma once

#include "../utils/basic-includes.h"
#include "../utils/utils.h"
#include "ProcessingUnity.h"


struct Filter {
	size_t _row;
	size_t _col;

	Filter(size_t row, size_t col) : _row(row), _col(col) { }
};

struct Padding {
	size_t _row;
	size_t _col;

	Padding(size_t row = 0, size_t col = 0) : _row(row), _col(col) { }
};




class ConvolutionCell : public IProcessingUnit {
	
	public:
		Eigen::MatrixXd _filter; // for DEBUG ONLY, must be private

	private:
		double _learningRate;
		double _bias;
		//Eigen::MatrixXd _filter;
		Eigen::MatrixXd _receivedInput;

		Padding _paddingSize;


	public:
		//ConvolutionCell(size_t filterSize, double learnRate = 0.01);
		ConvolutionCell(size_t filterRow, size_t filterCol, double learnRate = 0.001);
		ConvolutionCell(Filter filterSize, double learnRate = 0.001);
		ConvolutionCell(Filter filterSize, Padding padding, double learnRate = 0.001);
		~ConvolutionCell();

		static Eigen::MatrixXd Convolute(Eigen::MatrixXd& input, Eigen::MatrixXd& filter);
		static Eigen::MatrixXd Convolute(Eigen::MatrixXd& input, Eigen::MatrixXd& filter, size_t padding);
		static Eigen::MatrixXd Convolute(Eigen::MatrixXd& input, Eigen::MatrixXd& filter, size_t rowPadding, size_t colPadding);


		// Inherited via IProcessingUnit
		Eigen::MatrixXd Forward(Eigen::MatrixXd& input) override;
		Eigen::MatrixXd Backward(Eigen::MatrixXd& dLoss_dOutput) override;
		void UpdateLearningRate(size_t epoch, double error, std::function<void(size_t, double, double&)> UpdateRule) override;
};

