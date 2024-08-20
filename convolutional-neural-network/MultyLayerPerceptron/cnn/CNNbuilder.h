#pragma once

#include "../utils/basic-includes.h"
#include "cnn.h"

/*
class MlpBuilder {

	private:
	size_t inputSize;
	MLP _mlp;

	public:
	MlpBuilder();

	MLP Build();

	MlpBuilder InputSize(size_t size);
	MlpBuilder Architecture(std::vector<size_t> neuronsInLayer);
	MlpBuilder Architecture(std::vector<LayerSignature> layerSignature);
	MlpBuilder LostFunction(ILostFunction* lostFunction);
	MlpBuilder MaxEpochs(size_t epochs);
	MlpBuilder AcceptableAccuracy(double accuracy);
	MlpBuilder ParseLabelToVector(std::function<std::vector<double>(size_t)> CallBack);
	MlpBuilder SaveOn(std::string outFile);
	MlpBuilder LoadArchitectureFromJson(std::string file);
	MlpBuilder WhenToUpdateLearningRate(std::function<bool(size_t, double)> Conddition);
	MlpBuilder HowToUpdateLearningRate(std::function<double(size_t, double, double)> func);

};



	USE FORWARD TO DETERMIN OUTPUT MATRIX SIZE (DENSE LATER INPUT SIZE)


*/



class CNNbuilder {

	private:
		size_t inputSize;
		CNN _cnn;

		  
	public:
		CNNbuilder();

		CNN Build();

		CNNbuilder InputSize(size_t size);
		CNNbuilder Architecture(std::vector<size_t> neuronsInLayer);
		CNNbuilder Architecture(std::vector<DenseLayer> layerSignature);
		CNNbuilder LostFunction(ILostFunction* lossFunction);
		CNNbuilder MaxEpochs(size_t epochs);
		//CNNbuilder SaveOn(std::string outFile);
		//CNNbuilder LoadArchitectureFromJson(std::string file);

};
