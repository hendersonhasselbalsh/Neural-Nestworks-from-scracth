#pragma once


#include "../basic-includes.h"
#include "../02-Dense-Layer/DenseLayer.h" 
#include "../03-Activation-Function/ActivationFunction.h"
#include "../05-Loss-Function/LossFunctions.h"
#include "../06-Managers/DataManager.h"
#include "../06-Managers/LayerManager.h"
#include "../04-MLP/MLP.h"



class MLPbuilder {
	private:
		MLP _mlp;
		size_t _inputSize;

	public:
		MLPbuilder();
		MLPbuilder InputSize(size_t inputSize);
		MLPbuilder LossFunction(ILossFunction* lossFunc);
		MLPbuilder MaxEpochs(size_t maxEpochs);
		MLPbuilder BatchSize(long batchSize);
		MLPbuilder OutputClasses(size_t classes);
		MLPbuilder Architecture(std::vector<ILayer*> layers);

		MLP Build();
};