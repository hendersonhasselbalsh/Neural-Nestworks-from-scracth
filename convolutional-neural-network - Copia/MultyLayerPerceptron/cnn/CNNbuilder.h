#pragma once

#include "../utils/basic-includes.h"
#include "cnn.h"


class CNNbuilder {

	private:
		size_t _inputRow;
		size_t _inputCol;

		std::vector<DenseLayer> _denseLayerSignature;
		ILostFunction* _lossFunction;

		CNN _cnn;

		  
	public:
		CNNbuilder();

		CNN Build();

		CNNbuilder InputSize(size_t inputRow, size_t inputCol);
		CNNbuilder ProcessingArchitecture(std::vector<IProcessingUnit*> processingUnits);
		CNNbuilder DenseArchitecture(std::vector<DenseLayer> layerSignature);
		CNNbuilder LostFunction(ILostFunction* lossFunction);
		CNNbuilder MaxEpochs(size_t epochs);
		CNNbuilder ChangeLerningRate(std::function<void(size_t, double, double&)> func);    // epoch, error, learnrate

};
