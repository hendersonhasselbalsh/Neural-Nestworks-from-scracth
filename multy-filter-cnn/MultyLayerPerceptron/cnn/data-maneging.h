#pragma once

#include "../utils/basic-includes.h"
#include "../utils/utils.h"
#include "ProcessingUnity.h"


//--------------
// SCALE
//--------------

//class Scale : public IProcessingUnit {
//
//	private:
//		double _rangeStart;
//		double _rangeEnd;
//
//		double _max;
//		double _min;
//
//	public:
//		//ConvolutionCell(size_t poolSize, double learnRate = 0.01);
//		Scale(double rangeStart, double rangeEnd);
//		~Scale();
//
//
//		// Inherited via IProcessingUnit
//		Eigen::MatrixXd Forward(Eigen::MatrixXd& input) override;
//		Eigen::MatrixXd Backward(Eigen::MatrixXd& dLoss_dOutput) override;
//
//};




//--------------
// NORMALIZE
//--------------

//class Normalize : public IProcessingUnit {
//
//	private:
//		double _mean;
//		double _std_dev;
//
//	public:
//		Normalize();
//		~Normalize();
//
//
//		// Inherited via IProcessingUnit
//		/*Eigen::MatrixXd Forward(Eigen::MatrixXd& input) override;
//		Eigen::MatrixXd Backward(Eigen::MatrixXd& dLoss_dOutput) override;*/
//
//
//		static double Mean(Eigen::MatrixXd& matrix);
//		static double StandartDeviation(Eigen::MatrixXd& matrix);
//
//
//		// Inherited via IProcessingUnit
//		Eigen::Tensor<double, 3> Forward(Eigen::Tensor<double, 3>& input) override;
//		Eigen::Tensor<double, 3> Backward(Eigen::Tensor<double, 3>& dLoss_dOutput) override;
//
//};



//--------------
// NORMALIZATION
//--------------

class Normalization : public IProcessingUnit {

	private:
		std::vector<double> _layerStddev;
		std::vector<double> _layerMeans;

		double _betas;    // layer shift
		double _gammas;   // layer scala

		double _learningRate;

		Eigen::Tensor<double,3> _receivecInput;


	public:
		Normalization(double learningRate = 0.001);
		~Normalization();


		Eigen::MatrixXd LayerNorm(Eigen::MatrixXd& input, size_t inputIndex);

		// Inherited via IProcessingUnit
		Eigen::Tensor<double, 3> Forward(Eigen::Tensor<double, 3>& input) override;
		Eigen::Tensor<double, 3> Backward(Eigen::Tensor<double, 3>& dLoss_dOutput) override;



		Eigen::MatrixXd DLoss_DstdDev(Eigen::MatrixXd& dLoss_dy, size_t index);
		Eigen::MatrixXd DLoss_DVar(Eigen::MatrixXd& dLoss_dStdDev, size_t index);
		Eigen::MatrixXd DLoss_DNii(Eigen::MatrixXd& dLoss_dVar, size_t index);
		Eigen::MatrixXd DLoss_Dx(Eigen::MatrixXd& dLoss_dNii);

		Eigen::MatrixXd DLoss_DInput(Eigen::MatrixXd& dLoss_dy, size_t index);

};



