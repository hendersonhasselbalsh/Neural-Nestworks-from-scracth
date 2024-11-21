#include "Softmax.h"

Softmax::Softmax()
{
}


Eigen::MatrixXd Softmax::Activation(Eigen::MatrixXd& weitedSumVec)
{
    double sum = 0.0;
    for (size_t i = 0; i < weitedSumVec.rows(); i++) {
        sum += std::exp( weitedSumVec(i,0) );
    }

    size_t i = 0;
    Eigen::MatrixXd activationVec = weitedSumVec;

    for (auto weightedSum : weitedSumVec.rowwise()) {
        double U = weightedSum.value();
        activationVec(i++,0) = std::exp(U) / sum;
    }

    return activationVec;
}


Eigen::MatrixXd Softmax::dActivation_dWeightedSum(Eigen::MatrixXd& weitedSumVec)
{
    // not uses in this particular case
    return Eigen::MatrixXd();
}


Eigen::MatrixXd Softmax::Forward(Eigen::MatrixXd& weightedSumBatch)
{
    _activatedBatch = weightedSumBatch;

    for (auto& [weightedSumVec, vecIndex] : DataManager::ExtractVectors(weightedSumBatch)) {
        _activatedBatch.col(vecIndex) = Softmax::Activation(weightedSumVec);
    }

    return _activatedBatch;
}

Eigen::MatrixXd Softmax::Backward(Eigen::MatrixXd& dLdA_batch)
{
    Eigen::MatrixXd ones = Eigen::MatrixXd::Ones(dLdA_batch.rows(), dLdA_batch.cols());

    Eigen::MatrixXd dA_dU = _activatedBatch.cwiseProduct( ones -  _activatedBatch);
    Eigen::MatrixXd dL_dU = dLdA_batch.cwiseProduct(dA_dU);

    return dL_dU;
}
