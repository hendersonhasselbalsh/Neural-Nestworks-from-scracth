#pragma once

#include "../utils/basic-includes.h"
#include "../utils/utils.h"
#include "ProcessingUnity.h"


//--------------
// SCALE
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




//--------------
// NORMALIZE
//--------------

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



//--------------
// NORMALIZATION
//--------------

class Normalization : public IProcessingUnit {

	private:
		double _layerMeans;
		double _layerStddev;

		double _betas;    // layer shift
		double _gammas;   // layer scala

		double _learningRate;

		Eigen::MatrixXd _receivecInput;


	public:
		Normalization(double learningRate = 0.001);
		~Normalization();


		Eigen::MatrixXd LayerNorm(Eigen::MatrixXd& input);

		Eigen::MatrixXd DLoss_DstdDev(Eigen::MatrixXd& dLoss_dy);
		Eigen::MatrixXd DLoss_DVar(Eigen::MatrixXd& dLoss_dStdDev);
		Eigen::MatrixXd DLoss_DNii(Eigen::MatrixXd& dLoss_dVar);
		Eigen::MatrixXd DLoss_Dx(Eigen::MatrixXd& dLoss_dNii);

		Eigen::MatrixXd DLoss_DInput(Eigen::MatrixXd& dLoss_dy);


		// Inherited via IProcessingUnit
		Eigen::MatrixXd Forward(Eigen::MatrixXd& input) override;
		Eigen::MatrixXd Backward(Eigen::MatrixXd& dL_dNormalized) override;

};



