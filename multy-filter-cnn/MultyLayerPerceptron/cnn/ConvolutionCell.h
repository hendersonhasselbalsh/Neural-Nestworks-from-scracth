#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
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
	
	private:
		Eigen::Tensor<double, 3> _filter;
		Padding _paddingSize;
		double _learningRate;
		double _bias;

		Eigen::Tensor<double, 3> _receivedInput;

		size_t _filterQnt;


	public:
		ConvolutionCell(size_t filterQnt, Filter filterSize, double learnRate = 0.001);
		ConvolutionCell(size_t filterQnt, Filter filterSize, Padding padding, double learnRate = 0.001);
		~ConvolutionCell();

		static Eigen::MatrixXd Convolute(Eigen::MatrixXd& input, Eigen::MatrixXd& filter);
		static Eigen::MatrixXd Convolute(Eigen::MatrixXd& input, Eigen::MatrixXd& filter, size_t padding);
		static Eigen::MatrixXd Convolute(Eigen::MatrixXd& input, Eigen::MatrixXd& filter, size_t rowPadding, size_t colPadding);


		// Inherited via IProcessingUnit
		Eigen::Tensor<double, 3> Forward(Eigen::Tensor<double, 3>& input) override;
		Eigen::Tensor<double, 3> Backward(Eigen::Tensor<double, 3>& dLoss_dOutput) override;
		void UpdateLearningRate(size_t epoch, double error, std::function<void(size_t, double, double&)> UpdateRule) override;
};

