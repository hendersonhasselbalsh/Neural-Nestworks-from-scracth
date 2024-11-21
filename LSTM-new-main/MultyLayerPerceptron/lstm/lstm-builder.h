#pragma once

#include "../utils/basic-includes.h"
#include "lstm.h"




class LSTMbuilder {

	private:
		LSTM _lstm;

		size_t _classes;
		size_t _internalVectorSize;
		double _learningRate;

		std::vector<LayerSignature> _forgetArchitecture;
		std::vector<LayerSignature> _inputArchitecture;
		std::vector<LayerSignature> _candidateArchitecture;
		std::vector<LayerSignature> _outputArchitecture;

	public:
		LSTMbuilder();
		~LSTMbuilder();

		LSTM Build();


		LSTMbuilder InputSize(size_t size);
		LSTMbuilder CellStateSize(size_t size);
		LSTMbuilder LearningRate(double rate);
		LSTMbuilder LossFunction(ILostFunction* func);
		LSTMbuilder OutputClasses(size_t classes);

		LSTMbuilder ForgetArchitecture(std::vector<LayerSignature> architect);
		LSTMbuilder InputArchitecture(std::vector<LayerSignature> architect);
		LSTMbuilder CandidateArchitecture(std::vector<LayerSignature> architect);
		LSTMbuilder OutputArchitecture(std::vector<LayerSignature> architect);

		LSTMbuilder ForgetArchitecture(std::vector<size_t> architect, double learningRate = 0.0);
		LSTMbuilder InputArchitecture(std::vector<size_t> architect, double learningRate = 0.0);
		LSTMbuilder CandidateArchitecture(std::vector<size_t> architect, double learningRate = 0.0);
		LSTMbuilder OutputArchitecture(std::vector<size_t> architect, double learningRate = 0.0);

};

