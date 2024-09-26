#include "CNNbuilder.h"


CNNbuilder::CNNbuilder()
{
	_inputRow = 0;
	_inputCol = 0;

	_cnn = CNN();

	_cnn._UpdateLeraningRate = [](size_t epoch, double error, double& learnRate) { };
}






CNN CNNbuilder::Build()
{
	Eigen::MatrixXd text_matrix = Eigen::MatrixXd::Ones(_inputRow, _inputCol);

	for (auto& unit : _cnn._processingUnits) {
		text_matrix = unit->Forward(text_matrix);
	}

	size_t denseInputSize  =  text_matrix.rows() * text_matrix.cols();

	_cnn._mlp  =  MlpBuilder()
		.InputSize( denseInputSize )
		.Architecture( _denseLayerSignature )
		.LostFunction( _lossFunction )
		.Build();

	_cnn._mlp.UpdateLeraningRate = _cnn._UpdateLeraningRate;

	return _cnn;
}









CNNbuilder CNNbuilder::InputSize(size_t inputRow, size_t inputCol)
{
	_inputRow = inputRow;
	_inputCol = inputCol;

	return (*this);
}


CNNbuilder CNNbuilder::ProcessingArchitecture(std::vector<IProcessingUnit*> processingUnits)
{
	_cnn._processingUnits = processingUnits;

	return (*this);
}


CNNbuilder CNNbuilder::DenseArchitecture(std::vector<DenseLayer> layerSignature)
{
	_denseLayerSignature = layerSignature;

	return (*this);
}


CNNbuilder CNNbuilder::LostFunction(ILostFunction* lossFunction)
{
	_lossFunction = lossFunction;

	return (*this);
}


CNNbuilder CNNbuilder::MaxEpochs(size_t epochs)
{
	_cnn._maxEpochs = epochs;

	return (*this);
}



CNNbuilder CNNbuilder::ChangeLerningRate(std::function<void(size_t, double, double&)> func)
{
	_cnn._UpdateLeraningRate = func;
	return (*this);
}



