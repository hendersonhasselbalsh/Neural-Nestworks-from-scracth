#include "comparations.h"


void Comparations::Plot_Chart(Charts charts)
{
	GnuplotWrap gnuplot;
	gnuplot.xRange("0", "15");
	gnuplot.yRange("-0.001", "1");
	gnuplot.Grid("1", "0.1");

	gnuplot << "plot  ";

	for (auto& [fileName, chartTitle] : charts) {
		gnuplot << "\'" << fileName << "\' using 1:2 w l title \"" << chartTitle << "\", ";
	}

	gnuplot << "\n";
}


void Comparations::Plot_ComparisonChart(const std::string& file1, const std::string& name1, const std::string& file2, const std::string& name2)
{
	GnuplotWrap gnuplot; 
	gnuplot.OutFile("..\\..\\.resources\\gnuplot-output\\comparison.dat");  
	gnuplot.xRange("0", "");
	gnuplot.yRange("-0.01", "1.05");
	gnuplot.Grid("1", "0.1");


	std::ifstream file_1(file1);
	std::ifstream file_2(file2);

	while (!file_1.eof() && !file_2.eof()) {
		size_t epoch;
		double file1Data = 0, file2_data = 0, _;

		if (file_1.good()) {
			file_1 >> epoch;
			file_1 >> file1Data;

			std::cout << epoch << " : " << file1Data << "\n";
		}

		if (file_2.good()) {
			file_2 >> _;
			file_2 >> file2_data;
		}

		gnuplot.out << epoch << " " << file1Data << " " << file2_data << "\n";
	}

	gnuplot.out.close();
	gnuplot << "plot \'..\\..\\.resources\\gnuplot-output\\comparison.dat\' using 1:2 w l title \"" << name1 << "\", ";
	gnuplot << "\'..\\..\\.resources\\gnuplot-output\\comparison.dat\' using 1:3 w l title \"" << name2 << "\" \n";
	gnuplot << " \n";
}


void Comparations::With_WithOut_Adam(std::vector<std::pair<Eigen::MatrixXd, size_t>>& trainingDatas, std::vector<std::pair<Eigen::MatrixXd, size_t>>& testingData)
{
	MLP mlp_with_Adam =  MLPbuilder()
							.InputSize(28*28)
							.BatchSize(128)
							.Architecture({
								new DenseLayer(128, 1e-4),
								new ReLU(),
								new Dropout(0.1),
								new DenseLayer(10, 1e-4),
							})
							.LossFunction(new SoftmaxEntropy)
							.MaxEpochs(50)
							.UseAdam(0.9)
							.Build();


	MLP mlp_without_Adam = MLPbuilder()
							.InputSize(28*28)
							.BatchSize(128)
							.Architecture({
								new DenseLayer(128, 1e-4),
								new ReLU(),
								new Dropout(0.1),
								new DenseLayer(10, 1e-4),
							})
							.LossFunction(new SoftmaxEntropy)
							.MaxEpochs(50)
							.Build();


	const std::string with_adam_output_file = "..\\..\\.resources\\gnuplot-output\\with-adam.dat";
	std::ofstream with_adam = std::ofstream(with_adam_output_file.c_str());

	size_t epoch = 0;
	mlp_with_Adam.Training(trainingDatas, [&]() {
		std::cout << "\n\n----------------------- with ADAM " << epoch << " -----------------------\n\n\n";
		double error = Evaluator::Eval_MLP(mlp_with_Adam, testingData);

		with_adam << epoch << " " << error << "\n";

		epoch++;
	});
	with_adam.close();

	const std::string no_adam_output_file = "..\\..\\.resources\\gnuplot-output\\no-adam.dat";
	std::ofstream no_adam = std::ofstream(no_adam_output_file.c_str());



	epoch = 0;
	mlp_without_Adam.Training(trainingDatas, [&]() {
		std::cout << "\n\n----------------------- no ADAM " << epoch << " -----------------------\n\n\n";
		double error = Evaluator::Eval_MLP(mlp_without_Adam, testingData);

		no_adam << epoch << " " << error << "\n";

		epoch++;
	});
	no_adam.close(); 

	Plot_Chart({
		{with_adam_output_file, "with-adam"},
		{no_adam_output_file, "no-adam"},
	});

}


void Comparations::BatchSize(std::vector<std::pair<Eigen::MatrixXd, size_t>>& trainingDatas, std::vector<std::pair<Eigen::MatrixXd, size_t>>& testingData)
{
	std::vector<size_t> batches = { 1, 64, 128, 256, 512, 1024, 2048 };
	Charts charts;
	
	for (auto& batchSize : batches) {
		MLP mlp = MLPbuilder()
						.InputSize(28*28)
						.BatchSize(batchSize)
						.Architecture({
							new DenseLayer(20, 1e-5),
							new ReLU(),
		
							new Dropout(0.1),

							new DenseLayer(10, 1e-5),
						})
						.LossFunction(new SoftmaxEntropy)
						.MaxEpochs(30)
						.Build();
		

		std::stringstream ss;
		ss << "batch-" << batchSize;

		const std::string fileDir = std::string("..\\..\\.resources\\gnuplot-output\\") + ss.str() + ".dat";
		std::ofstream file = std::ofstream(fileDir.c_str()); 


		size_t epoch = 0;
		mlp.Training(trainingDatas, [&]() { 
			std::cout << "\n\n----------------------- " << ss.str() << ": " << epoch << " -----------------------\n\n\n";
			double error = Evaluator::Eval_MLP(mlp, testingData);
		
			file << epoch << " " << error << "\n";
		
			epoch++;
		});
		file.close();

		charts.push_back({fileDir, ss.str() });
	}


	Plot_Chart(charts);
}


void Comparations::LossFunction(std::vector<std::pair<Eigen::MatrixXd, size_t>>& trainingDatas, std::vector<std::pair<Eigen::MatrixXd, size_t>>& testingData)
{
	std::vector<std::pair<ILossFunction*, std::string>> lossFuncs = { 
		{ new MSE, "MSE" },
		{ new SoftmaxEntropy, "SoftmaxEntropy" },
	};

	Charts charts;


	for (auto& [lossFunc, funcName] : lossFuncs) {
		MLP mlp = MLPbuilder()
				.InputSize(28*28)
				.BatchSize(128)
				.Architecture({
					new DenseLayer(20, 1e-5),
					new ReLU(),

					new Dropout(0.1),

					new DenseLayer(10, 1e-5),
				})
				.LossFunction(lossFunc)
				.MaxEpochs(30)
				.Build();

		const std::string fileDir = std::string("..\\..\\.resources\\gnuplot-output\\") + funcName + ".dat";
		std::ofstream file = std::ofstream(fileDir.c_str());


		size_t epoch = 0;
		mlp.Training(trainingDatas, [&]() {
			std::cout << "\n\n----------------------- " << funcName << ": " << epoch << " -----------------------\n\n\n";
			double error = Evaluator::Eval_MLP(mlp, testingData);

			file << epoch << " " << error << "\n";

			epoch++;
		});
		file.close();

		charts.push_back({ fileDir, funcName });
	}


	Plot_Chart(charts);
}


void Comparations::ActivationFunction(std::vector<std::pair<Eigen::MatrixXd, size_t>>& trainingDatas, std::vector<std::pair<Eigen::MatrixXd, size_t>>& testingData)
{
	std::vector<std::pair<ILayer*, std::string>> actFuncs = {  
		{ new Sigmoid, "Sigmoid" },
		{ new Tanh, "Tanh" },
		{ new ReLU, "ReLU" },
	};

	Charts charts;


	for (auto [activationFunc, funcName] : actFuncs) {
		MLP mlp = MLPbuilder()
			.InputSize(28*28)
			.BatchSize(128)
			.Architecture({
				new DenseLayer(20, 1e-4),

				activationFunc,

				new Dropout(0.1),

				new DenseLayer(10, 1e-4),
			})
			.LossFunction(new SoftmaxEntropy)
			.MaxEpochs(30)
			.Build();

		const std::string fileDir = std::string("..\\..\\.resources\\gnuplot-output\\") + funcName + ".dat";
		std::ofstream file = std::ofstream(fileDir.c_str());


		size_t epoch = 0;
		mlp.Training(trainingDatas, [&]() {
			std::cout << "\n\n----------------------- " << funcName << ": " << epoch << " -----------------------\n\n\n";
			double error = Evaluator::Eval_MLP(mlp, testingData);

			file << epoch << " " << error << "\n";

			epoch++;
		});
		file.close();

		charts.push_back({ fileDir, funcName });
	}


	Plot_Chart(charts);
}


void Comparations::Adam_beta(std::vector<std::pair<Eigen::MatrixXd, size_t>>& trainingDatas, std::vector<std::pair<Eigen::MatrixXd, size_t>>& testingData)
{
	std::vector<double> betas = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, };

	Charts charts;


	for (auto& beta : betas) {
		/*MLP mlp = MLPbuilder()
					.InputSize(28*28)
					.BatchSize(128)
					.Architecture({
						new DenseLayer(20, 1e-5),
						new ReLU(),
						new Dropout(0.1),
						new DenseLayer(10, 1e-5),
					})
					.LossFunction(new SoftmaxEntropy)
					.MaxEpochs(30)
					.UseAdam(beta)
					.Build();*/

		std::stringstream ss;
		ss << "beta-" << beta;

		const std::string fileDir = std::string("..\\..\\.resources\\gnuplot-output\\") + ss.str() + ".dat";
		//std::ofstream file = std::ofstream(fileDir.c_str());


		/*size_t epoch = 0;
		mlp.Training(trainingDatas, [&]() {
			std::cout << "\n\n----------------------- " << ss.str() << ": " << epoch << " -----------------------\n\n\n";
			double error = Evaluator::Eval_MLP(mlp, testingData);

			file << epoch << " " << error << "\n";

			epoch++;
		});
		file.close();*/

		charts.push_back({ fileDir, ss.str() }); 
	}


	Plot_Chart(charts);
}
