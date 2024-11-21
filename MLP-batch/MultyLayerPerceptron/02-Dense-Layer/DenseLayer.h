#pragma once

#include "../basic-includes.h"
#include "../01-interfaces/ILayer.h"
#include "../06-Managers/DataManager.h"

class MLP;



class DenseLayer : public ILayer {
	private:
		Eigen::MatrixXd _weights;					// the neuron weights, each row of this matrix represent a neuron, the columns is represent indivitual weight of each neuron
		double _learningRate; 
		Eigen::MatrixXd _receivedInputBatch; 
		Eigen::MatrixXd _prev_dLdW;

	public:
		DenseLayer(size_t neuronQnt, double learnRate = 0.001);
		~DenseLayer();

		// Inherited via ILayer
		virtual Eigen::MatrixXd Forward(Eigen::MatrixXd& inputBatch) override;
		virtual Eigen::MatrixXd Backward(Eigen::MatrixXd& dL_dU) override;
		virtual void Initialize(__In__ size_t inputVecSize, __Out__ size_t* outputSize = nullptr) override;
		Eigen::MatrixXd AdamBackprop(Eigen::MatrixXd& dL_dU, double beta) override;

		friend class MLP;
};
