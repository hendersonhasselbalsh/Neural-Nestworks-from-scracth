#include "../utils/basic-includes.h"


namespace Utils {

	void PointwiseAdd(std::vector<double>* v1, std::vector<double>* v2, std::vector<double>* result);
	void PointwiseMult(std::vector<double>* v1, std::vector<double>* v2, std::vector<double>* result);
	std::vector<double> PointwiseTanh(std::vector<double>& v1);

	//------------------------
	std::map<std::pair<int, int>, int> getPairFrequencies(const std::vector<int>& data);
	void mergePair(std::vector<int>& data, const std::pair<int, int>& pair, int newToken);
	std::vector<int> bpeEncode(const std::vector<int>& data, int vocabSize);

	// Para leitura do dataset em texto
	std::vector<int> encodeFile(const std::string& filename, int vocabSize);
	std::vector<int> encode(const std::string& input);
	std::string resolveToken(int token);
	std::string decode(const std::vector<int>& encodedData);
}