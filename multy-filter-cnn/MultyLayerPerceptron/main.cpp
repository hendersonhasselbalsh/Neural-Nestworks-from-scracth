#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "gnuplot-include.h"
#include "utils/basic-includes.h"
#include "mlp/multy-layer-perceptron.h"
#include "cnn/cnn.h"




/*
    accuracy:  0.9855
    CNN cnn  =  CNNbuilder()
                    .InputSize(28,28)
                    .ProcessingArchitecture({
                        new ConvolutionCell(Filter{3,3}, 0.001),
                        new ActivationCell(new LeakyReLU()), 
                        new AveragePool(2,2),
                        new Normalize(),
                    })
                    .DenseArchitecture({
                        DenseLayer(256, new ReLU(), 0.001),
                        DenseLayer(10, new Sigmoid(), 0.001),
                    })
                    .LostFunction( new MSE() )
                    .MaxEpochs(80)
                    //.ChangeLerningRate( DecreaseLearningRate )
                    .Build();


*/



std::vector<CNN_DATA> LoadData_CNN(const std::string& folderPath)
{
    std::vector<CNN_DATA> set;

    int l = -1;

    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        if (std::filesystem::is_regular_file(entry.path())) {

            std::string fileName = entry.path().filename().string();
            std::string labelStr = Utils::SplitString(fileName, "_")[0];
            size_t labelIndex = (size_t)std::stoi(labelStr);

            size_t outputSize = 10;
            auto label = std::vector<double>(outputSize, 0.0);
            label[labelIndex] = 1.0;

            std::string fullPathName = entry.path().string();
            Eigen::MatrixXd inputMat = Utils::ImageToMatrix(cv::imread(fullPathName));

            Eigen::TensorMap<Eigen::Tensor<double, 3>> input(inputMat.data(), 1, 28, 28);

            CNN_DATA cnnData = {input, labelIndex};   
            cnnData.label = label;

            set.push_back(cnnData);

            if (labelIndex != l) {
                l = labelIndex;
                std::cout << "load data: [" << (labelIndex+1)*10 << "%]\n";
            }
        }
    }

    return set;
};

Eigen::MatrixXd TestingModelAccuracy(CNN* cnn, std::vector<CNN_DATA> testSet, double* accuracy)  // "..\\..\\.resources\\test"
{
    Eigen::MatrixXd confusionMatrix = Eigen::MatrixXd::Zero(10, 10);
    int totalData = 0;
    int errors = 0;

    for (auto& testData : testSet) {

        std::vector<double> givenOutput = cnn->Forward(testData.input);

        auto it = std::max_element(givenOutput.begin(), givenOutput.end());
        size_t givenLabel = std::distance(givenOutput.begin(), it);

        confusionMatrix(givenLabel, testData.labelIndex) += 1.0;

        totalData++;

        if (givenLabel != testData.labelIndex) { errors++; }
        
    }

    (*accuracy) = 1.0 - ((double)errors/totalData);

    return confusionMatrix;
}

void DecreaseLearningRate(size_t epoch, double error, double& learnRate)
{
    std::cout << "\nmodel accuracy: " << 1.0 - error << "\n";
    if (1.0-error >= 0.965) {
        learnRate  =  0.95*learnRate;
    }
}


int main(int argc, const char** argv)
{
    //--- initialize gnuplot to plot chart
    Gnuplot gnuplot;
    gnuplot.OutFile("..\\..\\.resources\\gnuplot-output\\res.dat");
    gnuplot.xRange("0", "");
    gnuplot.yRange("-0.01", "1.05");
    gnuplot.Grid("2", "0.1");

    //--- load MNIST training set
    std::cout << "LOATING TRAINING SET:\n";
    std::vector<CNN_DATA> trainigDataSet  =  LoadData_CNN("..\\..\\.resources\\train");

    //--- load MNIST test set
    std::cout << "\n\nLOATING TEST SET:\n";
    std::vector<CNN_DATA> testDataSet  =  LoadData_CNN("..\\..\\.resources\\test");


    //--- build CNN
    CNN cnn  =  CNNbuilder()
                    .InputSize(28, 28)
                    .ProcessingArchitecture({
                        new ConvolutionCell(4, Filter{3,3}, 0.001),
                        new ActivationCell(new LeakyReLU()),
                        //new AveragePool(2,2),
                        //new Normalization(),
                    })
                    .DenseArchitecture({
                        DenseLayer(256, new ReLU(), 0.001),
                        DenseLayer(10, new Sigmoid(2.0), 0.001),
                    })
                    .LostFunction(new MSE())
                    .MaxEpochs(80)
                    //.ChangeLerningRate( DecreaseLearningRate )
                    .Build();


    //--- training 
    double bestAccuracy = 0.0;
    size_t epoch = 0;

    cnn.Training(trainigDataSet, [&]() {
        double trainingAccuracy = 0.0;
        double testAccuracy = 0.0;
        Eigen::MatrixXd trainingConfusionMatrix  =  TestingModelAccuracy(&cnn, trainigDataSet, &trainingAccuracy);
        Eigen::MatrixXd testConfusionMatrix  =  TestingModelAccuracy(&cnn, testDataSet, &testAccuracy);

        if (testAccuracy > bestAccuracy) { bestAccuracy = testAccuracy; }

        std::cout << "------------ Training Epoch: " << epoch << " ------------\n";
        std::cout << "Training Accuracy: " << trainingAccuracy << "\n\n";
        std::cout << trainingConfusionMatrix << "\n\n\n";
        std::cout << "Test Accuracy: " << testAccuracy << "\n\n";
        std::cout << testConfusionMatrix << "\n\n\n";
        //std::cout << "filter: \n" << dynamic_cast<ConvolutionCell*>(cnn._processingUnits[0])->_filter << "\n\n\n\n";

        gnuplot.out << epoch << " " << trainingAccuracy << " " << testAccuracy << "\n";

        epoch++;
    });


    //--- plot chart
    gnuplot.out.close();
    gnuplot << "plot \'..\\..\\.resources\\gnuplot-output\\res.dat\' using 1:2 w l title \"Training Accuracy\", ";
    gnuplot << "\'..\\..\\.resources\\gnuplot-output\\res.dat\' using 1:3 w l title \"Test Accuracy\" \n";
    gnuplot << "set terminal pngcairo enhanced \n set output \'..\\..\\.resources\\gnuplot-output\\accuracy.png\' \n";
    gnuplot << " \n";


    std::cout << "BEST ACCURACY: " << bestAccuracy << "\n";
    std::cout << "[SUCESSO!!!!!]\n";
    return 0;
}




















//        _____ ______   ___       ________
//        |\   _ \  _   \|\  \     |\   __  \
//        \ \  \\\__\ \  \ \  \    \ \  \|\  \
//        \ \  \\|__| \  \ \  \    \ \   ____\
//        \ \  \    \ \  \ \  \____\ \  \___|
//        \ \__\    \ \__\ \_______\ \__\
//        \|__|     \|__|\|_______|\|__|



std::vector<MLP_DATA> LoadData(const std::string& folderPath)
{
    std::vector<MLP_DATA> set;

    int l = -1;

    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        if (std::filesystem::is_regular_file(entry.path())) {

            std::string fileName = entry.path().filename().string();
            std::string labelStr = Utils::SplitString(fileName, "_")[0];
            size_t label = (size_t)std::stoi(labelStr);

            std::string fullPathName = entry.path().string();
            Eigen::MatrixXd imgMat = Utils::ImageToMatrix(cv::imread(fullPathName));

            std::vector<double> input  =  Utils::FlatMatrix(imgMat);

            set.push_back({ input, label });

            if (label != l) {
                l = label;
                std::cout << "load data: [" << (label+1)*10 << "%]\n";
            }
        }
    }

    return set;
};

Eigen::MatrixXd TestingModelAccuracy(MLP* mlp, std::vector<MLP_DATA> testSet, double* accuracy)  // "..\\..\\.resources\\test"
{
    Eigen::MatrixXd confusionMatrix = Eigen::MatrixXd::Zero(10, 10);
    int totalData = 0;
    int errors = 0;

    for (auto& testData : testSet) {

        std::vector<double> givenOutput = mlp->Classify(testData.input);

        auto it = std::max_element(givenOutput.begin(), givenOutput.end());
        size_t givenLabel = std::distance(givenOutput.begin(), it);

        confusionMatrix(givenLabel, testData.labelIndex) += 1.0;

        totalData++;

        if (givenLabel != testData.labelIndex) { errors++; }

    }

    (*accuracy) = 1.0 - ((double)errors/totalData);

    return confusionMatrix;
}

std::vector<double> ParseLabelToEspectedOutput(size_t l)
{
    auto label = std::vector<double>((size_t)10, 0.0);
    label[l] = 1.0;
    return label;
}



int ____main(int argc, const char** argv)
{
    //--- initialize gnuplot to plot chart
    Gnuplot gnuplot;
    gnuplot.OutFile("..\\..\\.resources\\gnuplot-output\\res.dat");
    gnuplot.xRange("0", "");
    gnuplot.yRange("-0.01","1.05");
    gnuplot.Grid("5", "0.1");


    //--- load MNIST training set
    std::cout << "LOATING TRAINING SET:\n";
    std::vector<MLP_DATA> trainigDataSet  =  LoadData("..\\..\\.resources\\train"); 

    ////--- load MNIST test set
    std::cout << "\n\nLOATING TEST SET:\n";
    std::vector<MLP_DATA> testDataSet  =  LoadData("..\\..\\.resources\\test");


    //--- build mlp architecture and hiperparam
    MLP mlp  =  MlpBuilder()
                    .InputSize(28*28)
                    .Architecture({
                        DenseLayer(256, new ReLU(), 0.001),
                        DenseLayer(10, new NormalizedTanh(), 0.001),
                    })
                    .LostFunction(new MSE())
                    .MaxEpochs(100)
                    .ParseLabelToVector( ParseLabelToEspectedOutput )
                    .SaveOn("..\\..\\.resources\\gnuplot-output\\mlp\\mlp.json")
                    .Build();


    //--- training model, and do a callback on each epoch
    double bestAccuracy = 0.0;
    int epoch = 0;

    mlp.Training(trainigDataSet, [&](){
        double trainingAccuracy = 0.0;
        double testAccuracy = 0.0;
        Eigen::MatrixXd trainingConfusionMatrix  =  TestingModelAccuracy(&mlp, trainigDataSet, &trainingAccuracy);
        Eigen::MatrixXd testConfusionMatrix  =  TestingModelAccuracy(&mlp, testDataSet, &testAccuracy);

        if (testAccuracy > bestAccuracy) { bestAccuracy = testAccuracy; }

        std::cout << "------------ Training Epoch: " << epoch << " ------------\n";
        std::cout << "Training Accuracy: " << trainingAccuracy << "\n\n";
        std::cout << trainingConfusionMatrix << "\n\n\n";
        std::cout << "Test Accuracy: " << testAccuracy << "\n\n";
        std::cout << testConfusionMatrix << "\n\n\n\n";

        gnuplot.out << epoch << " " << trainingAccuracy << " " << testAccuracy << "\n";

        epoch++;
    });


    //--- plot chart
    gnuplot.out.close();
    gnuplot << "plot \'..\\..\\.resources\\gnuplot-output\\res.dat\' using 1:2 w l title \"Training Accuracy\", ";
    gnuplot << "\'..\\..\\.resources\\gnuplot-output\\res.dat\' using 1:3 w l title \"Test Accuracy\" \n";
    gnuplot << "set terminal pngcairo enhanced \n set output \'..\\..\\.resources\\gnuplot-output\\accuracy.png\' \n";
    gnuplot << " \n";


    std::cout << "BEST ACCURACY: " << bestAccuracy << "\n";
    std::cout << "[SUCESSO!!!!!]\n";

    return 0;
}

