#include "MLP.h"

MLP::MLP()
{
    _max_epochs = 0;
    _batchSize = 0;
}


Eigen::MatrixXd MLP::CalculateOutput(Eigen::MatrixXd& inputs)
{
    Eigen::MatrixXd output = inputs;
    for (auto layer : _layers) {
        if ( LayerManager::IsDenseLayer(layer) ) {
            output = LayerManager::PrepareVectorAsDenseLayerInput(output); 
        }
        output = layer->Forward(output);
    }

    return output;
}


Eigen::MatrixXd MLP::Backpropgagation(Eigen::MatrixXd& predictedY, Eigen::MatrixXd& correctY)
{
    Eigen::MatrixXd dL_dY = _lossFunc->dLoss_dpredictedY(predictedY, correctY);
    Eigen::MatrixXd dL_dX = Backpropgagation(dL_dY);
    return dL_dX;
}


Eigen::MatrixXd MLP::Backpropgagation(Eigen::MatrixXd& dL_dY)
{
    long lastLayerIndex = _layers.size() - 1;
    Eigen::MatrixXd dL_dO = dL_dY;

    for (long i = lastLayerIndex; i >= 0; i--) { 
        dL_dO = _layers[i]->Backward(dL_dO);
    }

    return dL_dO; // partial loss with respect to input;
}

void MLP::Training(std::vector<std::pair<Eigen::MatrixXd, size_t>>& datas, std::function<void(void)> callback)
{
    size_t epoch = 0;
    while (epoch++ < _max_epochs) {

        for (auto& [batchInputs, bachCorrectYs] : DataManager::BuildBatch(datas, _batchSize, _outputClasses)) {
            Eigen::MatrixXd predictedOutputs = MLP::CalculateOutput(batchInputs);
            Eigen::MatrixXd dL_dbatchXs = MLP::Backpropgagation(predictedOutputs, bachCorrectYs);
        }

        callback();
    }
}
