#include "LossFunctions.h"


//----------------------------
//  MEAN SQUARE ERROR
//----------------------------

double MSE::f(Eigen::MatrixXd& predictedY, Eigen::MatrixXd& correctY)
{
    assert(predictedY.size() == correctY.size());

    double H = (double)predictedY.size();
    double error = 0.0;

    for (size_t i = 0; i < predictedY.size(); i++) {
        double y = predictedY(i,0);
        double t = correctY(i, 0);

        error += std::pow(y - t, 2.0);
    }

    return (error / H);
}

Eigen::MatrixXd MSE::df(Eigen::MatrixXd& predictedY, Eigen::MatrixXd& correctY)
{
    assert(predictedY.size() == correctY.size());

    Eigen::MatrixXd dL_dY = Eigen::MatrixXd(predictedY.rows(), 1);

    for (size_t i = 0; i < predictedY.size(); i++) {
        double y = predictedY(i, 0);
        double t = correctY(i, 0);

        dL_dY(i,0) = 2.0 * (y - t);
    }

    return dL_dY;
}


double MSE::Loss(Eigen::MatrixXd& predictedY, Eigen::MatrixXd& correctY)
{
    double batchError = 0.0;
    double vecsQnt = predictedY.cols();

    for (auto& [predicted, correct] : DataManager::ExtractCorrespondingVectors(predictedY, correctY)) {
        double error = MSE::f(predicted, correct);
        batchError += error;
    }

    return (batchError/vecsQnt);
}

Eigen::MatrixXd MSE::dLoss_dpredictedY(Eigen::MatrixXd& predictedY, Eigen::MatrixXd& correctY)
{
    Eigen::MatrixXd dL_dY = Eigen::MatrixXd(predictedY.rows(), predictedY.cols());
    size_t vecIndex = 0;

    for (auto& [predicted, correct] : DataManager::ExtractCorrespondingVectors(predictedY, correctY)) {
        dL_dY.col(vecIndex++) = MSE::df(predicted, correct);
    }

    return dL_dY;
}


