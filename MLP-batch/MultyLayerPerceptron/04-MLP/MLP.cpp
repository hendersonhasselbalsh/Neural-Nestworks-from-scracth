#include "MLP.h"

MLP::MLP()
{
    max_epochs = 0;
    batchSize = 0;
}


bool MLP::IsDenseLayer(ILayer* layer)
{
    if (DenseLayer* d_ptr = dynamic_cast<DenseLayer*>(layer)) { 
        //cout << "basePtr is an instance of Derived class" <<endl;
        return true;
    } else {
        //cout << "basePtr is not ..." <<endl;
        return false;
    }
}

Eigen::MatrixXd MLP::CalculateOutput(Eigen::MatrixXd& inputs)
{
    Eigen::MatrixXd output = inputs;
    for (auto layer : _layers) {
        if ( IsDenseLayer(layer) ) { output = DataManager::PrepareVectorAsDenseLayerInput(output); }
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
        Eigen::MatrixXd dL_dInput = _layers[i]->Backward(dL_dO);
        dL_dO = dL_dInput;
    }

    return dL_dO; // partial loss with respect to input;
}

void MLP::Training(std::vector<std::pair<Eigen::MatrixXd, size_t>> data, std::function<void(void)> callback)
{
}
