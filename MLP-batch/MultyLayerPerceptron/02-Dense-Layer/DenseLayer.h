#pragma once

#include "../basic-includes.h"
#include "../01-interfaces/ILayer.h"
#include "DataManager.h"


class DenseLayer : public ILayer {
	public:
		Eigen::MatrixXd _weights;					// the neuron weights, each row of this matrix represent a neuron, the columns is represent indivitual weight of each neuron
		double _learningRate;						// 
		Eigen::MatrixXd _receivedInputBatch; 			//

	public:
		DenseLayer(size_t neuronQnt, double learnRate = 0.001);
		~DenseLayer();

		// Inherited via ILayer
		void Initialize(__In__ size_t inputVecSize, __Out__ size_t* outputSize = nullptr) override;
		virtual Eigen::MatrixXd Forward(Eigen::MatrixXd& inputBatch) override;
		virtual Eigen::MatrixXd Backward(Eigen::MatrixXd& dL_dU) override;

};
