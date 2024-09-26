#pragma once

#include "../utils/basic-includes.h"
#include "../utils/utils.h"
#include "ProcessingUnity.h"


//--------------
// MAX POOLING
//--------------

class MaxPool : public IProcessingUnit {

	private:
		Eigen::MatrixXd _dLoss_dPool;
		size_t _poolRow;
		size_t _poolCol;

	public:
		//ConvolutionCell(size_t poolSize, double learnRate = 0.01);
		MaxPool(size_t poolRow, size_t poolCol);
		~MaxPool();


		// Inherited via IProcessingUnit
		Eigen::MatrixXd Forward(Eigen::MatrixXd& input) override;
		Eigen::MatrixXd Backward(Eigen::MatrixXd& dLoss_dOutput) override;

};




//--------------
// MIN POOLING
//--------------

class MinPool : public IProcessingUnit {

	private:
	Eigen::MatrixXd _dLoss_dPool;
	size_t _poolRow;
	size_t _poolCol;

	public:
		//ConvolutionCell(size_t poolSize, double learnRate = 0.01);
	MinPool(size_t poolRow, size_t poolCol);
	~MinPool();


	// Inherited via IProcessingUnit
	Eigen::MatrixXd Forward(Eigen::MatrixXd& input) override;
	Eigen::MatrixXd Backward(Eigen::MatrixXd& dLoss_dOutput) override;

};




//--------------
// AVERAGE POOLING
//--------------

class AveragePool : public IProcessingUnit {

	private:
		size_t _poolRow;
		size_t _poolCol;

		double _size;

		size_t _inputRow;
		size_t _inputCol;

	public:
		//ConvolutionCell(size_t poolSize, double learnRate = 0.01);
		AveragePool(size_t poolRow, size_t poolCol);
		~AveragePool();


		// Inherited via IProcessingUnit
		Eigen::MatrixXd Forward(Eigen::MatrixXd& input) override;
		Eigen::MatrixXd Backward(Eigen::MatrixXd& dLoss_dOutput) override;

};

