#include "comparations.h"


void Comparations::Plot_ComparisonChart(const std::string& file1, const std::string& name1, const std::string& file2, const std::string& name2)
{
	Gnuplot gnuplot; 
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
			file_1 >> _;
			file_1 >> file1Data;

			std::cout << epoch << " : " << file1Data << "\n";
		}

		if (file_2.good()) {
			file_2 >> _;
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
	MLP mlp_with_Adam = MLPbuilder()
							.InputSize(28*28)
							.BatchSize(64)
							.Architecture({
								new DenseLayer(128, 0.0001),
								new ReLU(),
								new Dropout(0.3),

								new DenseLayer(64, 0.0001),
								new ReLU(),
								new Dropout(0.1),

								new DenseLayer(10, 0.0001),
							})
							.LossFunction(new SoftmaxEntropy)
							.MaxEpochs(40)
							.UseAdam(0.9)
							.Build();


	MLP mlp_without_Adam = MLPbuilder()
							.InputSize(28*28)
							.BatchSize(64)
							.Architecture({
								new DenseLayer(128, 0.0001),
								new ReLU(),
								new Dropout(0.3),

								new DenseLayer(64, 0.0001),
								new ReLU(),
								new Dropout(0.1),

								new DenseLayer(10, 0.0001),
							})
							.LossFunction(new SoftmaxEntropy)
							.MaxEpochs(40)
							.Build();


	const std::string with_adam_output_file = "..\\..\\.resources\\gnuplot-output\\with-adam.dat";
	const std::string no_adam_output_file = "..\\..\\.resources\\gnuplot-output\\no-adam.dat";

	std::ofstream with_adam = std::ofstream(with_adam_output_file.c_str());
	std::ofstream no_adam = std::ofstream(no_adam_output_file.c_str()); 


	size_t epoch = 0;
	mlp_with_Adam.Training(trainingDatas, [&]() {
		std::cout << "\n\n----------------------- with ADAM " << epoch << " -----------------------\n\n\n";
		double error = Evaluator::Eval_MLP(mlp_with_Adam, testingData);

		with_adam << epoch << " " << error;

		epoch++;
	});


	epoch = 0;
	mlp_without_Adam.Training(trainingDatas, [&]() {
		std::cout << "\n\n----------------------- no ADAM " << epoch << " -----------------------\n\n\n";
		double error = Evaluator::Eval_MLP(mlp_without_Adam, testingData);

		no_adam << epoch << " " << error;

		epoch++;
	});


	with_adam.close();
	no_adam.close(); 
	

	Plot_ComparisonChart(
		"..\\..\\.resources\\gnuplot-output\\with-adam.dat",
		"with-adam",
		"..\\..\\.resources\\gnuplot-output\\no-adam.dat",
		"no-adam"
	);


}
