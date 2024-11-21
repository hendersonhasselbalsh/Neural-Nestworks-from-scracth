#include "LayerManager.h"

bool LayerManager::IsDenseLayer(ILayer* layer)
{
    if (DenseLayer* d_ptr = dynamic_cast<DenseLayer*>(layer)) {
        //cout << "basePtr is an instance of Derived class" <<endl;
        return true;
    } else {
        //cout << "basePtr is not ..." <<endl;
        return false;
    }
}


bool LayerManager::IsBatchNorm(ILayer* layer)
{
    if (LayerNorm* d_ptr = dynamic_cast<LayerNorm*>(layer)) { 
        //cout << "basePtr is an instance of Derived class" <<endl;
        return true;
    } else {
        //cout << "basePtr is not ..." <<endl;
        return false;
    }
}

bool LayerManager::IsOptimizationLayer(ILayer* layer)
{
    if (IOptimizer* d_ptr = dynamic_cast<IOptimizer*>(layer)) {
        //cout << "basePtr is an instance of Derived class" <<endl;
        return true;
    } else {
        //cout << "basePtr is not ..." <<endl;
        return false;
    }
}


Eigen::MatrixXd LayerManager::PrepareVectorAsDenseLayerInput(Eigen::MatrixXd& vec)
{
    Eigen::MatrixXd input = Eigen::MatrixXd::Ones(vec.rows()+1, vec.cols());
    input.block(1, 0, vec.rows(), vec.cols()) = vec;

    return input;
}
