#include "../basic-includes.h"

#include "../02-Dense-Layer/DenseLayer.h"
#include "../03-Activation-Function/ActivationFunction.h"
#include "../04-MLP/MLP.h"
#include "DataLoader.h"
#include "../08-Optimizers/Optimizers.h"

using MNIST_DATA = std::vector<std::pair<Eigen::MatrixXd, size_t>>;



int main(int argc, const char** argv)
{
	/*
	DenseLayer denseLayer = DenseLayer(10, 0.0001);
	std::cout << denseLayer._weights << "\n\n\n";

	denseLayer.Initialize(16);
	std::cout << denseLayer._weights << "\n\n\n";

	Eigen::MatrixXd input(3,3);
	input << 
		1, 2, 3, 
		4, 5, 6, 
		7, 8, 9;

	denseLayer.Forward(input);
	std::cout << "\n\n\n\ninput\n" << (*denseLayer._receivedInputBatch) << "\n";
	/**/

	/*
	Eigen::MatrixXd input(3, 3);
	input <<
		1, 2, 3,
		4, 5, 6,
		7, 8, 9;

	Eigen::MatrixXd input2(3, 3);
	input2 <<
		1, 2, 3,
		4, 5, 6,
		7, 8, 9;

	for (auto& [dldU, dldW] : DataManager::ExtractCorrespondingVectors(input, input2)) {
		std::cout << "dldU\n" <<  dldU << "\n\n";
		std::cout << "dldW\n" <<  dldW.transpose() << "\n\n\n";
		Eigen::MatrixXd result = dldU * dldW.transpose();
		std::cout << "result\n" << result << "\n\n\n\n\n";
		std::cout << "\n-----------------------------------------------\n\n\n";
	}
	/**/

	/*
	Eigen::MatrixXd input(4, 3);
	input <<
		  1,   1,   1,
		0.1, 0.2, 0.3,
		0.4, 0.5, 0.6,
		0.7, 0.8, 0.9;

	DenseLayer l1 = DenseLayer(10, 0.0001);
	l1.Initialize(3);

	Sigmoid sigmoid = Sigmoid();
	sigmoid.Initialize(0);

	DenseLayer l3 = DenseLayer(4, 0.0001); 
	l3.Initialize(10);


	Eigen::MatrixXd o1 = l1.Forward(input);
		//Eigen::MatrixXd O1 = Eigen::MatrixXd::Ones(o1.rows()+1, o1.cols()); 
		//O1.block(1,0, o1.rows(), o1.cols()) = o1;
	std::cout << "o1:\n" << o1 << "\n\n\n";

	Eigen::MatrixXd o2 = sigmoid.Forward(o1);
		Eigen::MatrixXd O2 = Eigen::MatrixXd::Ones(o2.rows()+1, o2.cols());
		O2.block(1, 0, o2.rows(), o2.cols()) = o2;
	std::cout << "o2:\n" << O2 << "\n\n\n";

	Eigen::MatrixXd o3 = l3.Forward(O2); 
		Eigen::MatrixXd O3 = Eigen::MatrixXd::Ones(o3.rows()+1, o3.cols());
		O3.block(1, 0, o3.rows(), o3.cols()) = o3;
	std::cout << "o3:\n" << O3 << "\n\n\n";


	std::cout << *l1._receivedInputBatch <<"\n\n\n";
	std::cout << *sigmoid._activatedBach <<"\n\n\n";
	std::cout << *l3._receivedInputBatch <<"\n\n\n";
	/**/

	/*
	Eigen::MatrixXd inputBatch(4, 3);
	inputBatch <<
		0.3, 0.1, 0.5,
		0.1, 0.2, 0.3,
		0.4, 0.5, 0.6,
		0.7, 0.8, 0.9;

	Eigen::MatrixXd correbatchOutput(3,3);
	correbatchOutput << 
		1, 0, 0,
		0, 1, 0,
		0, 0, 1;


	MLP mlp = MLP();
	mlp._lossFunc = new MSE();

	mlp._layers = {
		new DenseLayer(10, 0.01),
		new Sigmoid(),
		new DenseLayer(3, 0.01),
	};


	size_t inputSize = 4;
	for (auto layer : mlp._layers) {
		size_t outputSize = 0;
		layer->Initialize(inputSize, &outputSize);
		inputSize = outputSize;
	}

	
	size_t epoch = 0;
	Eigen::MatrixXd predicted;

	while (epoch++ < 5000) {
		predicted = mlp.CalculateOutput(inputBatch);
		mlp.Backpropgagation(predicted, correbatchOutput);

		std::cout << "predicted:\n" << predicted << "\n\n";
	}
	/**/
	
	/*
	Eigen::MatrixXd I1 = Eigen::MatrixXd(5,1);
	I1 << 1, 2, 3, 4, 5;

	Eigen::MatrixXd I2 = Eigen::MatrixXd(5, 1);
	I2 << 6, 7, 8, 9, 10;

	Eigen::MatrixXd I3 = Eigen::MatrixXd(5, 1);
	I3 << 11, 12, 13, 14, 15;

	Eigen::MatrixXd I4 = Eigen::MatrixXd(5, 1);
	I4 << 16, 17, 18, 19, 20;

	Eigen::MatrixXd I5 = Eigen::MatrixXd(5, 1);
	I5 << 21, 22, 23, 24, 25;

	Eigen::MatrixXd I6 = Eigen::MatrixXd(5, 1);
	I6 << 26, 27, 28, 29, 30;


	std::vector<std::pair<Eigen::MatrixXd, size_t>> data ={
		{I1, 0},  
		{I2, 1}, 
		{I3, 2},
		{I4, 3},
		{I5, 4},
		{I6, 5},
	};


	//auto batchs = DataManager::BuildBatch(data, 5, 6);

	for (auto& [batchInput, correctYs] : DataManager::BuildBatch(data, 5, 6)) {
		std::cout << "batchInput\n" << batchInput << "\n\n";
		std::cout << "correctYs\n" << correctYs << "\n\n\n";
		std::cout << "\n-----------------------------------------------\n\n\n";
	}
	/**/

	/*
	Eigen::MatrixXd I1 = Eigen::MatrixXd(5,1);
	I1 << 0.01, 0.02, 0.03, 0.04, 0.0;

	Eigen::MatrixXd I2 = Eigen::MatrixXd(5, 1);
	I2 << 0.06, 0.07, 0.08, 0.0, 0.10;

	Eigen::MatrixXd I3 = Eigen::MatrixXd(5, 1);
	I3 << 0.11, 0.12, 0.0, 0.14, 0.15;

	Eigen::MatrixXd I4 = Eigen::MatrixXd(5, 1);
	I4 << 0.16, 0.0, 0.18, 0.19, 0.20;

	Eigen::MatrixXd I5 = Eigen::MatrixXd(5, 1);
	I5 << 0.0, 0.22, 0.23, 0.24, 0.25;

	Eigen::MatrixXd I6 = Eigen::MatrixXd(5, 1);
	I6 << 0.26, 0.27, 0.28, 0.29, 0.30;


	std::vector<std::pair<Eigen::MatrixXd, size_t>> datas ={
		{I1, 0},
		{I2, 1},
		{I3, 2},
		{I4, 3},
		{I5, 4},
		{I6, 5},
	};



	MLP mlp = MLP();
	mlp._batchSize = 3;
	mlp._lossFunc = new MSE();
	mlp._max_epochs = 1'000'000;
	mlp._outputClasses = 6;
	mlp._layers = {
		new DenseLayer(50, 0.01),
		new Sigmoid(),
		new DenseLayer(6, 0.01),
	};

	size_t inputSize = 5;
	for (auto& layer : mlp._layers) {
		size_t outputSize = 0;
		layer->Initialize(inputSize, &outputSize);
		inputSize = outputSize;
	}


	size_t epoch = 0;
	mlp.Training(datas, [&]() {
		Eigen::MatrixXd confusionMat = Eigen::MatrixXd::Zero(6,6);
		for (size_t i = 0; i < datas.size(); i++) {
			Eigen::MatrixXd output = mlp.CalculateOutput(datas[i].first);
			size_t predicted = 100, _;
			output.maxCoeff(&predicted, &_);  
			//confusionMat.col(i) = output;
			confusionMat(datas[i].second,predicted)++; 
		}

		if (epoch % 100 == 0) {
			std::cout << "\n\n---------------------------------------- " << epoch << " ----------------------------------------\n\n\n"; 
			std::cout << confusionMat << "\n\n\n";
		}

		epoch++;
	});
	/**/
	
	/*
	Eigen::MatrixXd I1 = Eigen::MatrixXd(5, 1);
	I1 << 0.01, 0.02, 0.03, 0.04, 0.0;

	Eigen::MatrixXd I2 = Eigen::MatrixXd(5, 1);
	I2 << 0.06, 0.07, 0.08, 0.0, 0.10;

	Eigen::MatrixXd I3 = Eigen::MatrixXd(5, 1);
	I3 << 0.11, 0.12, 0.0, 0.14, 0.15;

	Eigen::MatrixXd I4 = Eigen::MatrixXd(5, 1);
	I4 << 0.16, 0.0, 0.18, 0.19, 0.20;

	Eigen::MatrixXd I5 = Eigen::MatrixXd(5, 1);
	I5 << 0.0, 0.22, 0.23, 0.24, 0.25;

	Eigen::MatrixXd I6 = Eigen::MatrixXd(5, 1);
	I6 << 0.26, 0.27, 0.28, 0.29, 0.30;


	std::vector<std::pair<Eigen::MatrixXd, size_t>> datas ={
		{I1, 0},
		{I2, 1},
		{I3, 2},
		{I4, 3},
		{I5, 4},
		{I6, 5},
	};


	MLP mlp = MLPbuilder()
				.InputSize(5)
				.BatchSize(1)
				.Architecture({
					new DenseLayer(30, 0.01),
					new ReLU(),
					new DenseLayer(6, 0.01),
				})
				.LossFunction(new MSE)
				.MaxEpochs(10'000)
				.OutputClasses(6)
				.Build();


	size_t epoch = 0;
	mlp.Training(datas, [&](){
		Eigen::MatrixXd confusionMat = Eigen::MatrixXd::Zero(6,6);
		for (size_t i = 0; i < datas.size(); i++) {
			Eigen::MatrixXd output = mlp.CalculateOutput(datas[i].first);
			size_t predicted = 100, _;
			output.maxCoeff(&predicted, &_);
			//confusionMat.col(i) = output;
			confusionMat(datas[i].second, predicted)++;
		}

		if (epoch % 1000 == 0) {
			std::cout << "\n\n---------------------------------------- " << epoch << " ----------------------------------------\n\n\n";
			std::cout << confusionMat << "\n\n\n";
		}

		epoch++;
	});
	/**/

	
	MNIST_DATA traingDatas = DataLoader("..\\..\\.resources\\train").Load();
	MNIST_DATA testDatas = DataLoader("..\\..\\.resources\\test").Load();
	

	MLP mlp = MLPbuilder()
				.InputSize(28*28)
				.BatchSize(32)
				.Architecture({
					new DenseLayer(128, 0.001),
					new ReLU(),
					new Dropout(0.5),
					new DenseLayer(10, 0.001),
				})
				.LossFunction(new MSE)
				.MaxEpochs(40)
				.OutputClasses(10)
				.Build();


	size_t epoch = 0;

	mlp.Training(traingDatas, [&]() {
		std::cout << "\n\n\n---------------------------------------------- " << epoch << " ----------------------------------------------\n\n\n\n";
		
		std::cout << "TRAINING:\n";
		Evaluator::Eval_MLP(mlp, traingDatas);

		std::cout << "\n\n\TEST:\n";
		Evaluator::Eval_MLP(mlp, testDatas);
		
		epoch++;
	});


	std::cout << "\n\n\n[SUCCESS]!!!!!\n";
	return 0;
}
