#include "DenseLayer.h"


DenseLayer::DenseLayer(size_t neuronQnt, double learnRate)
{
    _weights = Eigen::MatrixXd::Zero(neuronQnt,1);
    _learningRate = learnRate;
}


DenseLayer::~DenseLayer()
{
}



void DenseLayer::Initialize(__In__ size_t inputSize, __Out__ size_t* outputSize)
{
    size_t neurons = _weights.rows();
    size_t weightSize = inputSize + 1;  // must include the bias
    
    _prev_dLdW = Eigen::MatrixXd::Zero(neurons, weightSize);
    _weights = Eigen::MatrixXd::Ones(neurons, weightSize);
    DataManager::XaviverInitialization(_weights, inputSize, neurons);

    (*outputSize) = neurons;
}


// inputBatch is all input vector from the batch, every column is a input vector
// weightedSum is the output of the dense layer,
// every weightedSums's column is a vector of weighted sum corresponding to the input vector
Eigen::MatrixXd DenseLayer::Forward(Eigen::MatrixXd& inputBatch)
{
    assert(_weights.cols() == inputBatch.rows() && "'_weights' must have one weight for each input's component");

    _receivedInputBatch = inputBatch;
    Eigen::MatrixXd weightedSums  =  _weights * inputBatch;
    return weightedSums;
}


Eigen::MatrixXd DenseLayer::Backward(Eigen::MatrixXd& dL_dU)
{
    // derivative with respect to the weight
    Eigen::MatrixXd dL_dW = Eigen::MatrixXd::Zero(_weights.rows(), _weights.cols());    // accumulated dLdW for each input in the batch

    // derivative with respect to the input
    Eigen::MatrixXd weightsWithoutBias = _weights.block(0, 1, _weights.rows(), _weights.cols()-1);
    Eigen::MatrixXd dL_dbatchX = Eigen::MatrixXd::Zero(weightsWithoutBias.cols(), _receivedInputBatch.cols());    // dLdX for each input X in the batch
    size_t vecIndex = 0;


    for (auto& [dLdU, dUdW] : DataManager::ExtractCorrespondingVectors(dL_dU, _receivedInputBatch)) {
        // derivative with respect to the weight
        Eigen::MatrixXd dLdW = dLdU * dUdW.transpose();
        dL_dW = dL_dW + dLdW;


        // derivative with respect to the input
        Eigen::MatrixXd dLdX = dLdU.transpose() * weightsWithoutBias;
        dL_dbatchX.col(vecIndex++) = dLdX.transpose();
    }
    
    _weights = _weights - _learningRate * dL_dW;

    return dL_dbatchX;
}


Eigen::MatrixXd DenseLayer::AdamBackprop(Eigen::MatrixXd& dL_dU, double beta)
{
    // derivative with respect to the weight
    Eigen::MatrixXd dL_dW = Eigen::MatrixXd::Zero(_weights.rows(), _weights.cols());    // accumulated dLdW for each input in the batch

    // derivative with respect to the input
    Eigen::MatrixXd weightsWithoutBias = _weights.block(0, 1, _weights.rows(), _weights.cols()-1);
    Eigen::MatrixXd dL_dbatchX = Eigen::MatrixXd::Zero(weightsWithoutBias.cols(), _receivedInputBatch.cols());    // dLdX for each input X in the batch
    size_t vecIndex = 0;


    for (auto& [dLdU, dUdW] : DataManager::ExtractCorrespondingVectors(dL_dU, _receivedInputBatch)) {
        // derivative with respect to the weight
        Eigen::MatrixXd dLdW = dLdU * dUdW.transpose();
        dL_dW = dL_dW + dLdW;


        // derivative with respect to the input
        Eigen::MatrixXd dLdX = dLdU.transpose() * weightsWithoutBias;
        dL_dbatchX.col(vecIndex++) = dLdX.transpose();
    }


    //Eigen::MatrixXd epson = Eigen::MatrixXd::Constant(_prev_dLdW.rows(), _prev_dLdW.cols(), 1e-8);
    //
    //_prev_dLdW = beta*_prev_dLdW + (1.0-beta)*dL_dW.cwiseProduct(dL_dW);
    //Eigen::MatrixXd inverse_prev_dLdW = _prev_dLdW.cwiseSqrt();
    //inverse_prev_dLdW = (_prev_dLdW + epson).cwiseInverse(); 
    //
    //_weights = _weights - _learningRate * (dL_dW.cwiseProduct(inverse_prev_dLdW)); 
    //_prev_dLdW = (beta*_prev_dLdW + (1.0-beta)*dL_dW);

    _weights = _weights - _learningRate * (beta*_prev_dLdW + (1.0-beta)*dL_dW);  
    _prev_dLdW = (beta*_prev_dLdW + (1.0-beta)*dL_dW);

    return dL_dbatchX;
}
