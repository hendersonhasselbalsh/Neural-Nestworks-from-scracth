#pragma once
#include "../utils/basic-includes.h"
#include "../utils/utils.h"
#include "ProcessingUnity.h"


class LayerNormalization : public IProcessingUnit {

	private:
		std::vector<double> _layerMeans;
		std::vector<double> _layerStddev;

		std::vector<double> _betas;    // layer shift
		std::vector<double> _gammas;   // layer scala

		double _learningRate;

		Eigen::MatrixXd _receivecInput;


	public:
		LayerNormalization(double learningRate = 0.001);
		~LayerNormalization();


		Eigen::MatrixXd LayerNorm(Eigen::MatrixXd& input);

		Eigen::MatrixXd DL_DVariance(Eigen::MatrixXd& dL_dNormalized);                               // horizontal vector
		Eigen::MatrixXd DL_DNii(Eigen::MatrixXd& dL_dVariance, Eigen::MatrixXd& dL_dNormalized);     // matrix, same dimention as input
		Eigen::MatrixXd DL_DMeans(Eigen::MatrixXd& dL_dNormalized);                                  // horizontal vector
		Eigen::MatrixXd DL_DInput(Eigen::MatrixXd& dL_dMean, Eigen::MatrixXd& dL_dNii);              // matrix, same dimention as input 


		// Inherited via IProcessingUnit
		Eigen::MatrixXd Forward(Eigen::MatrixXd& input) override;
		Eigen::MatrixXd Backward(Eigen::MatrixXd& dL_dNormalized) override;

};


