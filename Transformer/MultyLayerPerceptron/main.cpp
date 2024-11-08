#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "gnuplot-include.h"
#include "utils/basic-includes.h"
#include "tansformer/Encoder-Decoder-Transformer/Encoder-Decoder-Transformer.h"





Eigen::MatrixXd ConcatMatrix(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B)
{
    //assert(A.cols() == B.cols());
    Eigen::MatrixXd result;

    if (A.size() == 0) {
        result = B;
    }
    else {
        result  =  Eigen::MatrixXd(A.rows() + B.rows(), A.cols());

        result.block(0, 0, A.rows(), A.cols()) = A;
        result.block(A.rows(), 0, B.rows(), B.cols()) = B;
    }

    return result;
}


Eigen::MatrixXd GaneratedSentence(const Eigen::MatrixXd& sentence, const Eigen::MatrixXd& predictedToken)
{
    size_t maXIndice = 1000;
    Eigen::MatrixXd token  =  Eigen::MatrixXd::Zero(1, predictedToken.cols());
    predictedToken.row(0).maxCoeff(&maXIndice); 
    token(0, maXIndice) = 1.0;


    Eigen::MatrixXd newSentence = ConcatMatrix(sentence, token);

    return newSentence;
}







int main(int argc, const char** argv)  
{
    std::ofstream outputFile("..\\..\\.resources\\gnuplot-output\\transformer-output.txt"); 


    EncodeDecodeTransformer transformer  =  EncodeDecodeTransformer(64, 20, 1*2*2);       //EncodeDecodeTransformer(64, 20, 1*2*2); 



    Eigen::MatrixXd INPUT_WORDS = Eigen::MatrixXd(8, 20);
    INPUT_WORDS <<
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;



    Eigen::MatrixXd CORRECT_OUTPUT = Eigen::MatrixXd(7, 20);
    CORRECT_OUTPUT <<
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;



    Eigen::MatrixXd PREVIOUS_INDICIES = Eigen::MatrixXd::Zero(1, CORRECT_OUTPUT.rows()-1);

    size_t epoch = 0;
    bool correctPredictionNotFount = true;

    while (correctPredictionNotFount && epoch < 50'000) {

        Eigen::MatrixXd encoderInput  =  INPUT_WORDS;
        Eigen::MatrixXd decoderInput  = Eigen::MatrixXd(1, 20);
        decoderInput <<
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;


        Eigen::MatrixXd predictedSentence;

        for (size_t predictedWords = 0; predictedWords < CORRECT_OUTPUT.rows()-1; predictedWords++) {

            Eigen::MatrixXd predictedToken  =  transformer.Forward(encoderInput, decoderInput);

            predictedSentence  =  ConcatMatrix(predictedSentence, predictedToken);
            decoderInput  =  GaneratedSentence(decoderInput, predictedToken);
        }

        Eigen::MatrixXd expedtedSentence  =  CORRECT_OUTPUT.block(1, 0, CORRECT_OUTPUT.rows()-1, CORRECT_OUTPUT.cols());
        transformer.Backward(predictedSentence, expedtedSentence);


        //-----------------------------------------------------------------------------------------------
        //                  PRINT SENTENCE
        //-----------------------------------------------------------------------------------------------

        

        Eigen::MatrixXd CORRECT_INDICIES = Eigen::MatrixXd(1, CORRECT_OUTPUT.rows()-1);
        for (size_t row = 1; row < CORRECT_OUTPUT.rows(); row++) {
            size_t maXIndice = 0;
            CORRECT_OUTPUT.row(row).maxCoeff(&maXIndice);
            CORRECT_INDICIES(0, row-1)  =  maXIndice;
        }


        Eigen::MatrixXd PREDICTED_INDICIES = Eigen::MatrixXd(1, predictedSentence.rows());
        for (size_t row = 0; row < predictedSentence.rows(); row++) {
            size_t maXIndice = 0;
            predictedSentence.row(row).maxCoeff(&maXIndice);
            PREDICTED_INDICIES(0, row)  =  maXIndice;
        }

        std::cout << "--------------------------- epoch: " << epoch << " ---------------------------\n\n";
        std::cout << "CORRET:     " << CORRECT_INDICIES << "\n";
        std::cout << "PREDICTION: " << PREDICTED_INDICIES << "\n\n\n";
        

        // write output
        if ( PREVIOUS_INDICIES!=PREDICTED_INDICIES  &&  outputFile.is_open() ) {
            outputFile << "--------------------------- epoch: " << epoch << " ---------------------------\n\n";
            outputFile << "CORRET:     " << CORRECT_INDICIES << "\n";
            outputFile << "PREDICTION: " << PREDICTED_INDICIES << "\n\n\n";
            PREVIOUS_INDICIES = PREDICTED_INDICIES;
        }
        //-----------------------------------------------------------------------------------------------
        
        if (CORRECT_INDICIES == PREDICTED_INDICIES) { 
            correctPredictionNotFount = false; 
            std::cout <<  "\n\n\n\n0v0 CONGRATULATIONS CORRECT TRANSLATION\n\n\n\n";
        }

        epoch++;
    }




    outputFile.close();
    std::cout << "\n\n\n[DEBBUGED - SUCESSO!!!!]\n\n\n";
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



int ______main(int argc, const char** argv)
{
    //--- initialize gnuplot to plot chart
    std::cout << "TESTE updated backward\n\n";
    Gnuplot gnuplot;
    gnuplot.OutFile("..\\..\\.resources\\gnuplot-output\\res.dat");
    gnuplot.xRange("0", "");
    gnuplot.yRange("-0.01","1.05");
    gnuplot.Grid("5", "0.1");




    //--- load MNIST training set
    std::cout << "LOATING TRAINING SET:\n";
    std::vector<MLP_DATA> trainigDataSet  =  LoadData("..\\..\\.resources\\test");

    //--- load MNIST test set
    std::cout << "\n\nLOATING TEST SET:\n";
    std::vector<MLP_DATA> testDataSet  =  LoadData("..\\..\\.resources\\train-debug-8x8");




    //--- build mlp architecture and hiperparam
    MLP mlp  =  MlpBuilder()
                    .InputSize(28*28)
                    .Architecture({
                        DenseLayer(256, new ReLU(), 0.001), 
                        DenseLayer(10, new Sigmoid(2.0), 0.001, new MSE())
                    })
                    .LostFunction(new MSE())
                    .MaxEpochs(100)
                    .ParseLabelToVector( ParseLabelToEspectedOutput )
                    .SaveOn("..\\..\\.resources\\gnuplot-output\\mlp\\mlp.json")
                    .Build();




    //--- training model, and do a callback on each epoch
    double bestAccuracy = 0.0;
    size_t epoch = 0;

    mlp.Training(trainigDataSet, [&]() {
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

