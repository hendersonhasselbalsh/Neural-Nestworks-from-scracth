#include "gnuplot-include.h"
#include "lstm/lstm.h"
#include "mlp/multy-layer-perceptron.h"
#include "utils/basic-includes.h"
#include "utils/lstmUtils.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>



std::vector<TrainigData> LoadDataLSTM(const std::string& folderPath)
{
	//std::vector<MLP_DATA> set;
	std::vector<TrainigData> set;

	int l = -1;

	for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
		if (std::filesystem::is_regular_file(entry.path())) {

			std::string fileName = entry.path().filename().string();
			std::string labelStr = Utils::SplitString(fileName, "_")[0];
			//size_t label = (size_t)std::stoi(labelStr);    // MLP

			size_t labelIndex = (size_t)std::stoi(labelStr);                    // LSTM
			std::vector<double> label = std::vector<double>((size_t)10, 0.0);   // LSTM
			label[labelIndex] = 1.0;                                          // LSTM

			std::string fullPathName = entry.path().string();
			Eigen::MatrixXd imgMat = Utils::ImageToMatrix(cv::imread(fullPathName));

			std::vector<double> input = Utils::FlatMatrix(imgMat);

			set.push_back({ input, label });

			if (labelIndex != l) {
				l = labelIndex;
				std::cout << "load data: [" << (labelIndex + 1) * 10 << "%]\n";
			}
		}
	}

	return set;
};

Eigen::MatrixXd TestingModelAccuracyLSTM(LSTM* lstm, std::string path, double* accuracy)  // "..\\..\\.resources\\test"
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

			std::vector<double> givenOutput = lstm->Foward(inputs);

			auto it = std::max_element(givenOutput.begin(), givenOutput.end());
			int givenLabel = std::distance(givenOutput.begin(), it);

			confusionMatrix(givenLabel, label) += 1.0;

			totalData++;

			if (givenLabel != label) { errors++; }
		}
	}

	(*accuracy) = 1.0 - ((double)errors / totalData);

	return confusionMatrix;
}

std::vector<double> softmax(const std::vector<double>& input)
{
	std::vector<double> output(input.size());
	double max_val = *std::max_element(input.begin(), input.end());
	double sum = 0.0;

	for (size_t i = 0; i < input.size(); ++i) {
		output[i] = std::exp(input[i] - max_val);
		sum += output[i];
	}

	if (sum > 0.0) {
		for (size_t i = 0; i < output.size(); ++i) {
			output[i] /= sum;
		}
	} else {
		std::fill(output.begin(), output.end(), 1.0 / input.size());
	}

	return output;
}

void sample(LSTM model, int n_chars, int vocab_size, double top_p = 0.9)
{
	//model.ResetInternalMemory();

	std::random_device dev;
	std::mt19937 rng(dev());

	int seed = rng() % vocab_size;  // Random in [0, vocab_size)

	for (int i = 0; i < n_chars; i++) {
		auto word = std::vector<double>(vocab_size, 0.0);
		word[seed] = 1.0;

		std::vector<double> predictedNextWord = model.Foward(word);

		std::vector<double> softmax_probs = softmax(predictedNextWord);

		std::vector<std::pair<double, int>> prob_index_pairs;
		for (int j = 0; j < vocab_size; ++j) {
			prob_index_pairs.emplace_back(softmax_probs[j], j);
		}

		std::sort(prob_index_pairs.begin(), prob_index_pairs.end(), std::greater<>());

		double cumulative_prob = 0.0;
		int cutoff_index = 0;
		for (; cutoff_index < prob_index_pairs.size(); ++cutoff_index) {
			cumulative_prob += prob_index_pairs[cutoff_index].first;
			if (cumulative_prob >= top_p) {
				break;
			}
		}

		std::vector<double> top_p_probs(cutoff_index + 1);
		std::vector<int> top_p_indices(cutoff_index + 1);
		for (int k = 0; k <= cutoff_index; ++k) {
			top_p_probs[k] = prob_index_pairs[k].first;
			top_p_indices[k] = prob_index_pairs[k].second;
		}

		double sum = std::accumulate(top_p_probs.begin(), top_p_probs.end(), 0.0);
		if (sum > 0.0) {
			for (double& prob : top_p_probs) {
				prob /= sum;
			}
		}

		// Perform sampling based on top-p probabilities
		std::discrete_distribution<> dist(top_p_probs.begin(), top_p_probs.end());
		int sampled_index = top_p_indices[dist(rng)];

		std::cout << Utils::decode({ sampled_index });

		seed = sampled_index;
	}
}

double CEloss(const std::vector<double>& predicted, const std::vector<double>& target)
{
	double loss = 0.0;
	const double epsilon = 1e-9;
	for (size_t i = 0; i < target.size(); ++i) {
		double p = std::min(std::max(predicted[i], epsilon), 1.0 - epsilon);
		loss -= target[i] * std::log(p) + (1.0 - target[i]) * std::log(1.0 - p);
	}
	return loss;
}



int main(int argc, const char** argv)
{
	//--- initialize gnuplot to plot chart
	Gnuplot gnuplot;
	gnuplot.OutFile("..\\..\\.resources\\gnuplot-output\\res.dat");
	gnuplot.xRange("0", "");
	gnuplot.yRange("-0.01", "1.05");
	gnuplot.Grid("1", "0.1");


	//--- load TEXT full dataset
	std::vector<int> dataset = Utils::encodeFile("..\\..\\.resources\\text.txt", 128 + 200);


	// -- Hyperparametros
	size_t vocab_size = 128 + 200;	// Tamanho do vocabul�rio
	size_t block_size = 256;	    // Qual o tamanho da sequ�ncia a ser considerada.
	int reps = 2048;				// Quantos blocos de treino por Epoch (Maior = maior cobertura da base de dados)
	int epochs = 250;

	//--- build LSTM
	LSTM lstm = LSTMbuilder()
					.InputSize(vocab_size)
					.ForgetArchitecture({
						LayerSignature(32, new Sigmoid(), 0.002)
					})
					.InputArchitecture({
						LayerSignature(32, new Sigmoid(), 0.002)
					})
					.CandidateArchitecture({
						LayerSignature(32, new Tanh(), 0.002)
					})
					.OutputArchitecture({
						LayerSignature(32, new Sigmoid(), 0.002)
					})
					.LearningRate(0.002)
					.LossFunction(new MSE())
					.OutputClasses(vocab_size)
					.Build();


	std::string train = "";
	std::string generated = "";

	//--- train
	size_t epoch = 0;
	while (epoch < epochs) {
		double runningLoss = 0.0;

		double currAcc = 0.0;

		train = "";
		generated = "";

		//--- train for this epoch
		for (size_t step = 0; step < reps; step++) {

			//lstm.ResetInternalMemory();

			if (step % 5 == 0)
				std::cout << "Epoch " << std::setw(4) << std::setfill(' ') << epoch
				<< " | Step: " << std::setw(4) << std::setfill(' ') << step
				<< " (" << std::setw(6) << std::fixed << std::setprecision(3) << 100. * (double)step / (double)reps << "%)"
				<< " | Loss: " << std::setw(7) << std::fixed << std::setprecision(3) << runningLoss / (double)(step * block_size)
				<< " | Running Accuracy: " << std::setw(6) << std::fixed << std::setprecision(2) << 100. * currAcc / (double)(step * block_size) << "%      \r";

			int i = rand() % (dataset.size() - block_size);

			for (int disp = 0; disp < block_size; disp++) {

				int input = dataset[(i + disp)];
				int next = dataset[(i + disp) + 1];

				auto word = std::vector<double>((size_t)vocab_size, 0.0);
				auto NextWord = std::vector<double>((size_t)vocab_size, 0.0);

				word[input] = 1.0;
				NextWord[next] = 1.0;

				std::vector<double> predictedNextWord = softmax(lstm.Foward(word));

				int pnw = std::distance(
					predictedNextWord.begin(),
					std::max_element(predictedNextWord.begin(), predictedNextWord.end())
				);

				runningLoss += CEloss(predictedNextWord, NextWord);

				lstm.Backward(predictedNextWord, NextWord);

				if (next == pnw) currAcc += 1;

				if (step == reps - 1) {
					train += Utils::decode({ next });
					generated += Utils::decode({ pnw });
				}
			}
		}

		std::cout << "------------------------------------ Epoch " << epoch << " ------------------------------------\n\n";
		std::cout << "C.E.Loss: " << runningLoss / (double)(block_size * reps) << std::endl;
		std::cout << "Accuracy: " << 100.0 * currAcc / (double)(reps * block_size) << "% " << std::endl;
		std::cout << "\nOriginals: " << train << std::endl;
		std::cout << "\nGenerated: " << generated << "\n\n";
		std::cout << "\nSample:\n\n";
		sample(lstm, 256, vocab_size);
		std::cout << "\n\n";

		gnuplot.out << epoch << " " << currAcc / (double)(reps * block_size) << "\n";

		epoch++;
	}



	//--- plot chart
	gnuplot.out.close();
	gnuplot << "plot \'..\\..\\.resources\\gnuplot-output\\res.dat\' using 1:2 w l title \"Training Accuracy\" \n";
	gnuplot << "set terminal pngcairo enhanced \n set output \'..\\..\\.resources\\gnuplot-output\\accuracy.png\' \n";
	gnuplot << " \n";


	std::cout << "[SUCESSO]!!!\n";

	return 0;
}
