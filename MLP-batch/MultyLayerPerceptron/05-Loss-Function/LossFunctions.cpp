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




//----------------------------
//  SOFTMAX ENTROPY
//----------------------------

double SoftmaxEntropy::f(Eigen::MatrixXd& predictedY, Eigen::MatrixXd& correctY)
{

    double error = 0.0;
    Eigen::MatrixXd probs = Softmax().Activation(predictedY); 

    for (size_t i = 0; i < probs.cols(); i++) {
        double p = probs(i,0);
        double t = correctY(i,0);

        error += -t * std::log(p);
    }
    
    return error; // no need to divide by size
}


Eigen::MatrixXd SoftmaxEntropy::df(Eigen::MatrixXd& predictedY, Eigen::MatrixXd& correctY)
{
    Eigen::MatrixXd probs = Softmax().Activation(predictedY); 
    Eigen::MatrixXd dL_dY = Eigen::MatrixXd(predictedY.rows(), 1);

    for (size_t i = 0; i < probs.rows(); i++) {
        double p = probs(i,0);
        double t = correctY(i,0);

        if (t == 1.0) { dL_dY(i, 0) = p - 1.0; }
        else /*t==0*/ { dL_dY(i, 0) = p; }
        
    }

    return dL_dY;
}


double SoftmaxEntropy::Loss(Eigen::MatrixXd& predictedY, Eigen::MatrixXd& correctY)
{
    double H = (double) predictedY.cols();
    double batchError = 0.0;

    for (auto& [predictVec, correctVec] : DataManager::ExtractCorrespondingVectors(predictedY, correctY)) {
        double error = SoftmaxEntropy::f(predictVec, correctVec);
        batchError += error;
    }

    return (batchError/H); 
}



Eigen::MatrixXd SoftmaxEntropy::dLoss_dpredictedY(Eigen::MatrixXd& predictedY, Eigen::MatrixXd& correctY)
{
    size_t vec = 0;
    Eigen::MatrixXd dL_dY = Eigen::MatrixXd(predictedY.rows(), predictedY.cols());

    for (auto& [predictVec, correctVec] : DataManager::ExtractCorrespondingVectors(predictedY, correctY)) {
        dL_dY.col(vec++) = SoftmaxEntropy::df(predictVec, correctVec);
    }

    return dL_dY;
}


/*
//----------------------------
//  CROSS ENTROPY
//----------------------------

double CrossEntropy::f(Eigen::MatrixXd& predictedY, Eigen::MatrixXd& correctY)
{

    double error = 0.0;

    for (size_t i = 0; i < predictedY.cols(); i++) {
        double y = predictedY(i, 0); 
        double t = correctY(i, 0);

        error += -t * std::log(y);
    }

    return error; // no need to divide by size
}


Eigen::MatrixXd CrossEntropy::df(Eigen::MatrixXd& predictedY, Eigen::MatrixXd& correctY)
{
    double EPSON = 1e-5;
    Eigen::MatrixXd dL_dY = Eigen::MatrixXd(predictedY.rows(), 1);

    for (size_t i = 0; i < predictedY.rows(); i++) { 
        double p = predictedY(i, 0); 
        double t = correctY(i, 0);

        if (p > 1.0) { p = 1.0; } 
        else if (p < 0.0) { p = 0.0001; }

        if (t == 1.0) { dL_dY(i, 0) = -1.0/(p); } 
        else /*t==0 { dL_dY(i, 0) = p; }

    }

    return dL_dY;
}


double CrossEntropy::Loss(Eigen::MatrixXd& predictedY, Eigen::MatrixXd& correctY)
{
    double H = (double)predictedY.cols();
    double batchError = 0.0;

    for (auto& [predictVec, correctVec] : DataManager::ExtractCorrespondingVectors(predictedY, correctY)) {
        double error = CrossEntropy::f(predictVec, correctVec);
        batchError += error;
    }

    return (batchError/H);
}


Eigen::MatrixXd CrossEntropy::dLoss_dpredictedY(Eigen::MatrixXd& predictedY, Eigen::MatrixXd& correctY)
{
    size_t vec = 0;
    Eigen::MatrixXd dL_dY = Eigen::MatrixXd(predictedY.rows(), predictedY.cols());

    for (auto& [predictVec, correctVec] : DataManager::ExtractCorrespondingVectors(predictedY, correctY)) {
        dL_dY.col(vec++) = CrossEntropy::df(predictVec, correctVec); 
    }

    return dL_dY;
}
*/
