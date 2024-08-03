#pragma once

#include "../utils/basic-includes.h"
#include "../utils/lstmUtils.h"
#include "../mlp/multy-layer-perceptron.h"

class LSTMbuilder;



class LSTM {

	private:
		std::vector<double> _previousCellState;
		std::vector<double> _cellState;
		std::vector<double> _hiddenState;

		MLP _forgetMLP;
		MLP _inputMLP;
		MLP _candidateMLP;
		MLP _outputMLP;

		MLP _linearMLP;

		std::vector<double> _gatesInput;

		std::vector<double> _forgetActivation;
		std::vector<double> _inputActivation;
		std::vector<double> _candidateActivation;
		std::vector<double> _outputActivation;

		std::vector<double> _linearOutput;

		std::vector<double> _lossWithRespectToHiddenState;

		LSTM();

	public:
		size_t _inputSize;
		size_t _hiddenStateSize;
		size_t _cellStateSaze;

		double _learningRate;
		ILostFunction* _lossFunc;


		~LSTM();



		std::vector<double> Foward( std::vector<double> input );
		void Backward(std::vector<double> predictedY, std::vector<double> correctY);
		

		std::vector<double> LossPartialWithRespectToLinearOutput(std::vector<double>& predictedY, std::vector<double>& correctY, std::vector<double>& dSoftmax);
		std::vector<double> LossPartialWithRespectToHiddenState(std::vector<double>& predictedY, std::vector<double>& correctY, std::vector<double>& dSoftmax);
		std::vector<double> LossPartialWithRespectToCellState(std::vector<double>& dLoss_dHiddenState);

		std::vector<double> LossPartialWithRespectToOutput(std::vector<double>& dLoss_dHiddenState);
		std::vector<double> LossPartialWithRespectToCandidate(std::vector<double>& dLoss_dCellState);
		std::vector<double> LossPartialWithRespectToInput(std::vector<double>& dLoss_dCellState);
		std::vector<double> LossPartialWithRespectToForget(std::vector<double>& dLoss_dCellState);
		

		std::vector<double> Softmax(const std::vector<double>& input);
		std::vector<double> dSoftmax(const std::vector<double>& input);

	friend class LSTMbuilder;
};


#include "lstm-builder.h"