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
    "bring",
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
    "lord",
    "lords",
    "men",
    "Mordor",
    "mortal",
    "nine",
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
    "as",
    "aprisionar",
    "ceu",
    "condenados",
    "de",
    "deitam",
    "do",
    "dominar",
    "dos",
    "e",
    "em",
    "encontrar",
    "encontra-los",
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
    "Mordor",
    "morrer",
    "mortais",
    "na",
    "nas",
    "nove",
    "rochosos",
    "seu",
    "sombrio",
    "sete",
    "saloes",
    "senhor",
    "senhores",
    "seus",
    "se",
    "sob",
    "sombras",
    "traze-los",
    "terras",
    "trazer",
    "trono",
    "todos",
    "tres",
    "um",
}; 


std::vector<std::string> EN_SENTENCES = {
    "<sos> three rings for the Elven kings under the sky <eos>",
    "<sos> seven for the Dwarf lords in their halls of stone <eos>",
    "<sos> nine for mortal men doomed to die <eos>",
    "<sos> one for the dark lord on his dark throne <eos>",
    "<sos> in the land of Mordor where the shadows lie <eos>",
    "<sos> one ring to rule them all <eos>",
    "<sos> one ring to find them <eos>",
    "<sos> one ring to bring them all <eos>",
    "<sos> and in the darkness bind them <eos>",
};

std::vector<std::string> PT_SENTENCES ={
    "<sos> tres aneis para os elfos reis sob o ceu <eos>",
    "<sos> sete aneis para os senhores anoes em seus saloes rochosos <eos>",
    "<sos> nove para os homens mortais condenados a morrer <eos>",
    "<sos> um para o senhor sombrio em seu trono sombrio <eos>",
    "<sos> nas terras de Mordor onde as sombras deitam <eos>",
    "<sos> um anel para todos governar <eos>",
    "<sos> um anel para todos encontrar <eos>",
    "<sos> um anel para todos trazer <eos>",
    "<sos> e na escuridao aprisionar <eos>",
};


std::string ORIGINAL_SENTENCE = "<sos> one ring to rule them all one ring to find them <eos>";
std::string CORRECT_TRANSLATION = "<sos> um anel para todos governar um anel para encontra-los <eos>";





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

    size_t index = 100'000'000;
    for (size_t i = 0; i < dictionarySize; i++) {
        if (word.compare( dictionary[i]) == 0) { index = i; }
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






int ___main(int argc, const char** argv)
{
    std::ofstream outputFile("..\\..\\.resources\\gnuplot-output\\transformer-output.txt");


    EncodeDecodeTransformer transformer  =  TransformerBuilder()
                                                .EmbeddingSize(64*2*2*2)
                                                .InputDictionarySize(EN_DICTIONARY.size())
                                                .OutputDictionarySize(PT_DICTIONARY.size())
                                                .Heads(1*2*2*2)
                                                .LearningRate(0.001)
                                                .Build();



    std::string PREVIOUS_TRANSLATION = "";


    size_t epoch = 0;

    while (epoch < 50'000) {
        //std::cout << "\n\n\n------------------------------------------------------ epoch: " << epoch << " ------------------------------------------------------\n\n";

        for (size_t i = 0; i < EN_SENTENCES.size(); i++) {
            Eigen::MatrixXd CORRECT_OUTPUT = SentenceToMatrix(PT_SENTENCES[i], PT_DICTIONARY);
            Eigen::MatrixXd encoderInput  =  SentenceToMatrix(EN_SENTENCES[i], EN_DICTIONARY);
            Eigen::MatrixXd decoderInput  = Eigen::MatrixXd::Zero(1, PT_DICTIONARY.size());
            decoderInput(0, 0) = 1.0;


            Eigen::MatrixXd predictedSentence;
            for (size_t predictedWords = 0; predictedWords < CORRECT_OUTPUT.rows()-1; predictedWords++) {

                Eigen::MatrixXd predictedToken  =  transformer.Forward(encoderInput, decoderInput);

                predictedSentence  =  ConcatMatrix(predictedSentence, predictedToken);
                Eigen::MatrixXd correctToken = CORRECT_OUTPUT.row(i+1);
                decoderInput  =  GaneratedSentence(decoderInput, correctToken);
            }

            Eigen::MatrixXd expedtedSentence  =  CORRECT_OUTPUT.block(1, 0, CORRECT_OUTPUT.rows()-1, CORRECT_OUTPUT.cols());
            transformer.Backward(predictedSentence, expedtedSentence);

            /*std::cout << "CORRECT SENTENCE:    " << MatrixToSentence(CORRECT_OUTPUT, PT_DICTIONARY) << "\n";
            std::cout << "TANSLATION:          " << MatrixToSentence(decoderInput, PT_DICTIONARY) << "\n\n\n";*/
        }

        //-----------------------------------------------------------------------------------------------
        //                  PRINT SENTENCE
        //-----------------------------------------------------------------------------------------------
        /*Eigen::MatrixXd encoderInput  =  SentenceToMatrix(ORIGINAL_SENTENCE, EN_DICTIONARY);
        Eigen::MatrixXd decoderInput  = Eigen::MatrixXd::Zero(1, PT_DICTIONARY.size());
        decoderInput(0, 0) = 1.0;
        for (size_t predictedWords = 0; predictedWords < 6; predictedWords++) {
            Eigen::MatrixXd predictedToken  =  transformer.Forward(encoderInput, decoderInput);
            decoderInput  =  GaneratedSentence(decoderInput, predictedToken);
        }

        std::cout << "\n\n\n------------------------------------------------------ epoch: " << epoch << " ------------------------------------------------------\n\n";
        std::cout << "ORIGINAL SENTENCE:    " << ORIGINAL_SENTENCE << "\n";
        std::cout << "TANSLATION:           " << MatrixToSentence(decoderInput, PT_DICTIONARY) << "\n";*/

        if (epoch % 10 == 0) {
            std::cout << "\n\n\n------------------------------------------------------ epoch: " << epoch << " ------------------------------------------------------\n\n";
            for (size_t i = 0; i < EN_SENTENCES.size(); i++) {
                Eigen::MatrixXd CORRECT_OUTPUT1 = SentenceToMatrix(PT_SENTENCES[i], PT_DICTIONARY);
                Eigen::MatrixXd encoderInput1  =  SentenceToMatrix(EN_SENTENCES[i], EN_DICTIONARY);
                Eigen::MatrixXd decoderInput1  = Eigen::MatrixXd::Zero(1, PT_DICTIONARY.size());
                decoderInput1(0, 0) = 1.0;


                Eigen::MatrixXd predictedSentence1;
                for (size_t predictedWords = 0; predictedWords < CORRECT_OUTPUT1.rows()-1; predictedWords++) {

                    Eigen::MatrixXd predictedToken1  =  transformer.Forward(encoderInput1, decoderInput1);

                    predictedSentence1  =  ConcatMatrix(predictedSentence1, predictedToken1);
                    decoderInput1  =  GaneratedSentence(decoderInput1, predictedToken1);
                }

                std::cout << "CORRECT SENTENCE:    " << MatrixToSentence(CORRECT_OUTPUT1, PT_DICTIONARY) << "\n";
                std::cout << "TANSLATION:          " << MatrixToSentence(decoderInput1, PT_DICTIONARY) << "\n\n\n";
            }
        }
        //-----------------------------------------------------------------------------------------------

        epoch++;
    }




    outputFile.close();
    std::cout << "\n\n\n[DEBBUGED - SUCESSO!!!!]\n\n\n";
    return 0;
}




int main(int argc, const char** argv)
{
    std::ofstream outputFile("..\\..\\.resources\\gnuplot-output\\transformer-output.txt");


    EncodeDecodeTransformer transformer  =  TransformerBuilder()
                                                .EmbeddingSize(64*2*2*2)
                                                .InputDictionarySize(EN_DICTIONARY.size())
                                                .OutputDictionarySize(PT_DICTIONARY.size())
                                                .Heads(1*2*2*2)
                                                .LearningRate(0.0001)
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
        if (PREVIOUS_TRANSLATION.compare(PREDICTED_TRANSLATION)!=0  &&  outputFile.is_open()) {
            outputFile << "--------------------------- iteration: " << epoch << " ---------------------------\n\n";
            outputFile << "ORIGINAL SENTENCE:    " << ORIGINAL_SENTENCE << "\n";
            outputFile << "CORRET TANSLATION:    " << CORRECT_TRANSLATION << "\n";
            outputFile << "PREDICTED TANSLATION: " << PREDICTED_TRANSLATION << "\n\n\n";
            PREVIOUS_TRANSLATION = PREDICTED_TRANSLATION;
        }
        //-----------------------------------------------------------------------------------------------

        if (PREDICTED_TRANSLATION.compare(CORRECT_TRANSLATION) == 0) {
            correctPredictionNotFount = false;
            std::cout <<  "\n\n\n\n\t\t (˶ᵔ ᵕ ᵔ˶) CONGRATULATIONS CORRECT TRANSLATION (˶ᵔ ᵕ ᵔ˶)\n\n\n\n";
        }

        epoch++;
    }




    outputFile.close();
    std::cout << "\n\n\n[DEBBUGED - SUCESSO!!!!]\n\n\n";
    return 0;
}


