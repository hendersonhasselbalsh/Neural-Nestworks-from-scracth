#include "MLPbuilder.h"

MLPbuilder::MLPbuilder()
{
	_mlp = MLP();
}


MLPbuilder MLPbuilder::InputSize(size_t inputSize)
{
	_inputSize = inputSize;
	return (*this);
}


MLPbuilder MLPbuilder::LossFunction(ILossFunction* lossFunc)
{
	_mlp._lossFunc = lossFunc;
	return (*this);
}


MLPbuilder MLPbuilder::MaxEpochs(size_t maxEpochs)
{
	_mlp._max_epochs = maxEpochs;
	return (*this);
}


MLPbuilder MLPbuilder::BatchSize(long batchSize)
{
	_mlp._batchSize = batchSize;
	return (*this);
}


MLPbuilder MLPbuilder::OutputClasses(size_t classes)
{
	_mlp._outputClasses = classes;
	return (*this);
}


MLPbuilder MLPbuilder::Architecture(std::vector<ILayer*> layers)
{
	_mlp._layers = layers;

	for (auto& layer : _mlp._layers) {
		size_t outputSize;
		layer->Initialize(_inputSize, &outputSize);
		_inputSize = outputSize;
	}

	return (*this);
}


MLP MLPbuilder::Build()
{
	return _mlp;
}
