#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "gnuplot-include.h"
#include "utils/basic-includes.h"
#include "mlp/multy-layer-perceptron.h"

#include "cnn/ConvolutionCell.h"




int main(int argc, const char** argv)
{
    cv::Mat img = cv::imread("..\\..\\.resources\\humano-original.png");
    Eigen::MatrixXd imgMatrix = Utils::ImageToMatrix(img);

    Eigen::MatrixXd filter = Eigen::MatrixXd(3,3);
    filter << 
        (1.0/9.0), (1.0/9.0), (1.0/9.0),
        (1.0/9.0), (1.0/9.0), (1.0/9.0),
        (1.0/9.0), (1.0/9.0), (1.0/9.0);

    Eigen::MatrixXd convMatrix = ConvolutionCell::Convolute(imgMatrix, filter);
    cv::Mat convolvedImg = Utils::MatrixToImage(convMatrix);


    cv::imshow("original", img);
    cv::imshow("convolved", convolvedImg);


    cv::waitKey(0);

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

Eigen::MatrixXd TestingModelAccuracy(MLP* mlp, std::string path, double* accuracy)  // "..\\..\\.resources\\test"
{
    Eigen::MatrixXd confusionMatrix = Eigen::MatrixXd::Zero(10, 10);
    int totalData = 0;
    int errors = 0;

    for (const auto& entry : std::filesystem::directory_iterator(path.c_str())) {
        if (std::filesystem::is_regular_file(entry.path())) {

            std::string fileName = entry.path().filename().string();
            std::string labelStr = Utils::SplitString(fileName, "_")[0];
            int label = std::stoi(labelStr);

            std::string fullPathName = entry.path().string();
            Eigen::MatrixXd input = Utils::ImageToMatrix(cv::imread(fullPathName));

            std::vector<double> inputs = Utils::FlatMatrix(input);

            std::vector<double> givenOutput = mlp->Classify(inputs);

            auto it = std::max_element(givenOutput.begin(), givenOutput.end());
            int givenLabel = std::distance(givenOutput.begin(), it);

            confusionMatrix(givenLabel, label) += 1.0;

            totalData++;

            if (givenLabel != label) { errors++; }
        }
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



int _main__MLP__(int argc, const char** argv)
{
    //--- initialize gnuplot to plot chart
    Gnuplot gnuplot;
    gnuplot.OutFile("..\\..\\.resources\\gnuplot-output\\res.dat");
    gnuplot.xRange("0", "");
    gnuplot.yRange("-0.01","1.05");
    gnuplot.Grid("5", "0.1");


    //--- load MNIST training set
    std::vector<MLP_DATA> trainigDataSet  =  LoadData("..\\..\\.resources\\train-debug-8x8" );


    //--- build mlp architecture and hiperparam
    MLP mlp  =  MlpBuilder()
                    .InputSize(28*28)
                    .Architecture({
                        DenseLayer(5, new Tanh(), 0.001),
                        DenseLayer(7, new Tanh(), 0.001),
                        DenseLayer(10, new Linear(), 0.001, new MSE())
                    })
                    .MaxEpochs(100)
                    .ParseLabelToVector( ParseLabelToEspectedOutput )
                    .SaveOn("..\\..\\.resources\\gnuplot-output\\mlp\\mlp.json")
                    .Build();


    //--- training model, and do a callback on each epoch
    int ephocCounter = -1;

    mlp.Training(trainigDataSet, [&mlp, &trainigDataSet, &ephocCounter, &gnuplot](){
        ephocCounter++;
        double accuracy = 0.0;

        Eigen::MatrixXd confusionMatrix  =  TestingModelAccuracy(&mlp, "..\\..\\.resources\\train-debug-8x8", &accuracy);

        std::cout << "Training Epoch: " << ephocCounter << "\n";
        std::cout << "Training Accuracy: " << accuracy << "\n\n";
        std::cout << confusionMatrix << "\n\n\n\n";

        gnuplot.out << ephocCounter << " " << accuracy << "\n";
    });


    //--- evaluating training on MNIST test set
    ////double accuracy = 0.0;
    ////Eigen::MatrixXd confusionMatrix  =  TestingModelAccuracy(&mlp, "..\\..\\.resources\\test", &accuracy);
    ////std::cout << "Testing Accuracy: " << accuracy << "\n\n";
    ////std::cout << confusionMatrix << "\n\n\n\n";


    //--- plot chart
    gnuplot.out.close();
    gnuplot << "plot \'..\\..\\.resources\\gnuplot-output\\res.dat\' using 1:2 w l title \"Training Accuracy\" \n";
    gnuplot << "set terminal pngcairo enhanced \n set output \'..\\..\\.resources\\gnuplot-output\\accuracy.png\' \n";
    gnuplot << " \n";


    std::cout << "\n\n[SUCESSO]";

    return 0;
}

