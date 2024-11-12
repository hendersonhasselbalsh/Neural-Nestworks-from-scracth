#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "gnuplot-include.h"
#include "utils/basic-includes.h"
#include "tansformer/Encoder-Decoder-Transformer/Encoder-Decoder-Transformer.h"
#include "TransformerBuilder.h"



std::vector<std::string> EN_DICTIONARY = {
    "<sos>",
    "<eos>",
    "all",
    "and",
    "bind",
    "dark",
    "darkness",
    "die",
    "doomed",
    "Dwarf",
    "Elven",
    "find",
    "for",
    "halls",
    "his",
    "in",
    "kings",
    "land",
    "and",
    "lie",
    "lords",
    "men",
    "Mordor",
    "mortal",
    "Nine",
    "of",
    "on",
    "one",
    "ring",
    "rings",
    "rule",
    "seven",
    "shadows",
    "sky",
    "stone",
    "the",
    "their",
    "them",
    "three",
    "throne",
    "to",
    "under",
    "where",
};

std::vector<std::string> PT_DICTIONARY = {
    "<sos>",
    "<eos>",
    "anoes",
    "anel",
    "aneis",
    "a",
    "aprisionar",
    "ceu",
    "de",
    "deitam",
    "do",
    "dominar",
    "dos",
    "em",
    "encontrar",
    "escuro",
    "escuridao",
    "fadados",
    "governar",
    "homens",
    "o",
    "os",
    "onde",
    "onde",
    "para",
    "reis",
    "elfos",
    "rochosos",
    "saloes",
    "senhores",
    "seus",
    "se",
    "sob",
    "sombras",
    "trazer",
    "todos",
    "tres",
    "um",
}; 


std::vector<std::string> EN_SENTENCES = {
    "<sos> three rings for the Elven kings under the sky <eos>",
    "<sos> seven for the Dwarf lords in their halls of stone <eos>",
    "<sos> Nine for Mortal Men doomed to die <eos>",
    "<sos> One for the Dark Lord on his dark throne <eos>",
};

std::vector<std::string> PT_SENTENCES ={
    "<sos> tres aneis para os elfos reis sob o ceu <eos>",
    "<sos> sete aneis para os senhores anoes em seus saloes rochosos <eos>",
    "<sos> nove para os homens mortais condenados a morrer <eos>",
    "<sos> um para o senhor sombrio em seu trono sombrio <eos>",
};


std::string ORIGINAL_SENTENCE = "<sos> one ring to rule them all <eos>";
std::string CORRECT_TRANSLATION = "<sos> um anel para todos governar <eos>";





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


Eigen::MatrixXd WordToToken(std::string& word, std::vector<std::string>& dictionary)
{
    size_t dictionarySize = dictionary.size();
    Eigen::MatrixXd token  =  Eigen::MatrixXd::Zero(1, dictionarySize);

    size_t index = 0;
    for (size_t i = 0; i < dictionarySize; i++) {
        if (word == dictionary[i]) { index = i; }
    }

    token(0,index) = 1.0;

    return token;
}

Eigen::MatrixXd SentenceToMatrix(std::string& sentence, std::vector<std::string>& dictionary)
{
    std::vector<std::string> sentenceWord  =  Utils::SplitString(sentence, " ");

    Eigen::MatrixXd sentenceMatrix;
    for (auto word : sentenceWord) {
        Eigen::MatrixXd token  =  WordToToken(word, dictionary);
        sentenceMatrix  =  ConcatMatrix(sentenceMatrix, token); 
    }

    return sentenceMatrix;
}


std::string MatrixToSentence(Eigen::MatrixXd& mat, std::vector<std::string>& dictionary)
{
    std::string sentence  =  "";

    for (size_t row = 0; row < mat.rows(); row++) {
        size_t maXIndice = 0;
        mat.row(row).maxCoeff(&maXIndice); 
        sentence  +=  dictionary[maXIndice]  +  " ";
    }

    return sentence;
}

/*
std::string PredictedSentence(Eigen::MatrixXd& predictedSentence, std::vector<std::string>& dictionary)
{
    Eigen::MatrixXd sos_token  =  Eigen::MatrixXd::Zero(1, predictedSentence.cols()); 
    sos_token(0, 0) = 1.0;

    Eigen::MatrixXd sentenceToken = ConcatMatrix(sos_token, predictedSentence);
    std::string sentence  =  MatrixToSentence(sentenceToken, dictionary);

    return sentence;
}
*/



int main(int argc, const char** argv)
{
    std::ofstream outputFile("..\\..\\.resources\\gnuplot-output\\transformer-output.txt");


    EncodeDecodeTransformer transformer  =  TransformerBuilder()
                                                .EmbeddingSize(64*2*2)
                                                .InputDictionarySize(EN_DICTIONARY.size())
                                                .OutputDictionarySize(PT_DICTIONARY.size())
                                                .Heads(1*2*2*2)
                                                .LearningRate(0.001)
                                                .Build();


    Eigen::MatrixXd INPUT_WORDS = SentenceToMatrix(ORIGINAL_SENTENCE, EN_DICTIONARY);
    Eigen::MatrixXd CORRECT_OUTPUT = SentenceToMatrix(CORRECT_TRANSLATION, PT_DICTIONARY);



    std::string PREVIOUS_TRANSLATION = "";


    size_t epoch = 0;
    bool correctPredictionNotFount = true;

    while (correctPredictionNotFount && epoch < 50'000) {

        Eigen::MatrixXd encoderInput  =  INPUT_WORDS;
        Eigen::MatrixXd decoderInput  = Eigen::MatrixXd::Zero(1, PT_DICTIONARY.size());
        decoderInput(0,0) = 1.0;


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
        //std::string PREDICTED_TRANSLATION = PredictedSentence(predictedSentence, PT_DICTIONARY);
        std::string PREDICTED_TRANSLATION = MatrixToSentence(decoderInput, PT_DICTIONARY);

        std::cout << "--------------------------- iteration: " << epoch << " ---------------------------\n\n";
        std::cout << "ORIGINAL SENTENCE:    " << ORIGINAL_SENTENCE << "\n";
        std::cout << "CORRET TANSLATION:    " << CORRECT_TRANSLATION << "\n";
        std::cout << "PREDICTED TANSLATION: " << PREDICTED_TRANSLATION << "\n\n\n";


        // write output
        if (PREVIOUS_TRANSLATION!=PREDICTED_TRANSLATION  &&  outputFile.is_open()) {
            outputFile << "--------------------------- iteration: " << epoch << " ---------------------------\n\n";
            outputFile << "ORIGINAL SENTENCE:    " << ORIGINAL_SENTENCE << "\n";
            outputFile << "CORRET TANSLATION:    " << CORRECT_TRANSLATION << "\n";
            outputFile << "PREDICTED TANSLATION: " << PREDICTED_TRANSLATION << "\n\n\n";
            PREVIOUS_TRANSLATION = PREDICTED_TRANSLATION;
        }
        //-----------------------------------------------------------------------------------------------

        if (PREDICTED_TRANSLATION == CORRECT_TRANSLATION) {
            correctPredictionNotFount = false;
            std::cout <<  "\n\n\n\n\t\t (˶ᵔ ᵕ ᵔ˶) CONGRATULATIONS CORRECT TRANSLATION (˶ᵔ ᵕ ᵔ˶)\n\n\n\n";
        }

        epoch++;
    }




    outputFile.close();
    std::cout << "\n\n\n[DEBBUGED - SUCESSO!!!!]\n\n\n";
    return 0;
}







int ____main(int argc, const char** argv)  
{
    std::ofstream outputFile("..\\..\\.resources\\gnuplot-output\\transformer-output.txt"); 

    std::cout << "teste 1\n";
    // EncodeDecodeTransformer(64*2, 20, 20, 1*2*2);   // EncodeDecodeTransformer(64, 20, 1*2*2); 
    EncodeDecodeTransformer transformer  =  TransformerBuilder()
                                                .EmbeddingSize(64*2*2*2)
                                                .InputDictionarySize(20)
                                                .OutputDictionarySize(20)
                                                .Heads(1*2*2*2)
                                                .Build();


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
            std::cout <<  "\n\n\n\n\t\t (˶ᵔ ᵕ ᵔ˶) CONGRATULATIONS CORRECT TRANSLATION (˶ᵔ ᵕ ᵔ˶)\n\n\n\n";
        }

        epoch++;
    }




    outputFile.close();
    std::cout << "\n\n\n[DEBBUGED - SUCESSO!!!!]\n\n\n";
    return 0;
}


