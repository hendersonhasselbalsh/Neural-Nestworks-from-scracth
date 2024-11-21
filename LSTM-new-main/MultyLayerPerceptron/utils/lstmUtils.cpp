#include "lstmUtils.h"



void Utils::PointwiseAdd(std::vector<double>* v1, std::vector<double>* v2, std::vector<double>* result)
{
	std::transform(v1->begin(), v1->end(), v2->begin(), result->begin(), std::plus<double>());
}



void Utils::PointwiseMult(std::vector<double>* v1, std::vector<double>* v2, std::vector<double>* result)
{
	std::transform(v1->begin(), v1->end(), v2->begin(), result->begin(), std::multiplies<double>());
}



std::vector<double> Utils::PointwiseTanh(std::vector<double>& v1)
{
	std::vector<double> result = std::vector<double>(v1.size(), 0.0);

	size_t i = 0;
	for (auto& x : v1) {
		result[i]  =  std::tanh( x );
		i++;
	}

	return result;
}


//--------------------------------------


std::map<std::pair<int, int>, int> Utils::getPairFrequencies(const std::vector<int>& data)
{
    std::map<std::pair<int, int>, int> pairFreqs;
    for (size_t i = 0; i < data.size() - 1; ++i) {
        std::pair<int, int> pair ={ data[i], data[i + 1] };
        pairFreqs[pair]++;
    }
    return pairFreqs;
}

void Utils::mergePair(std::vector<int>& data, const std::pair<int, int>& pair, int newToken)
{
    std::vector<int> newData;
    size_t i = 0;
    while (i < data.size()) {
        if (i < data.size() - 1 && data[i] == pair.first && data[i + 1] == pair.second) {
            newData.push_back(newToken);
            i += 2;
        } else {
            newData.push_back(data[i]);
            i++;
        }
    }
    data = newData;
}

std::map<int, std::pair<int, int> > bpeTokenMappings;

struct PairFrequency {
    std::pair<int, int> pair;
    int frequency;

    bool operator<(const PairFrequency& other) const
    {
        return frequency < other.frequency;
    }
};

std::vector<int> Utils::bpeEncode(const std::vector<int>& data, int vocabSize)
{
    std::vector<int> encodedData = data;
    int nextToken = 128;

    std::map<int, std::string> tokenToString;

    std::priority_queue<PairFrequency> maxHeap;

    auto pairFreqs = getPairFrequencies(encodedData);
    for (const auto& pairFreq : pairFreqs) {
        maxHeap.push(PairFrequency{ pairFreq.first, pairFreq.second });
    }

    while (nextToken < vocabSize && !maxHeap.empty()) {
        auto mostFrequentPair = maxHeap.top().pair;
        maxHeap.pop();

        std::cout << "\nComposite Token: " <<  nextToken - 128 << " -> \'" << decode({ mostFrequentPair.first, mostFrequentPair.second }) << "\'";

        bpeTokenMappings[nextToken] = mostFrequentPair;

        mergePair(encodedData, mostFrequentPair, nextToken);

        pairFreqs = getPairFrequencies(encodedData);
        maxHeap = std::priority_queue<PairFrequency>();
        for (const auto& pairFreq : pairFreqs) {
            maxHeap.push(PairFrequency{ pairFreq.first, pairFreq.second });
        }

        nextToken++;
    }

    return encodedData;
}

std::vector<int> Utils::encodeFile(const std::string& filename, int vocabSize)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Could not open file");

    std::cout << "Processing " << filename << "...\n";

    std::vector<int> encodedData;
    char ch;
    while (file.get(ch)) {
        encodedData.push_back(static_cast<unsigned char>(ch));
    }

    // Apply BPE to reduce the token count to vocab_size
    encodedData = Utils::bpeEncode(encodedData, vocabSize);

    std::cout << "\n\nProcessed.\n\n";

    return encodedData;
}

std::vector<int> Utils::encode(const std::string& input)
{
    return { 127 };
}

std::string Utils::resolveToken(int token)
{

    if (0 < token && token < 128) return std::string(1, static_cast<char>(token));

    auto it = bpeTokenMappings.find(token);

    if (it == bpeTokenMappings.end())
    {
        std::cout << "= Bad token! [" + std::to_string(token) + "] =";
        return "?";
    }

    const auto& mapping = it->second;
    std::string result;
    result += resolveToken(mapping.first);
    result += resolveToken(mapping.second);

    return result;
}



std::string Utils::decode(const std::vector<int>& encodedData)
{
    std::string decodedStr;
    for (int value : encodedData) {
        if (value >= 128)
            decodedStr += resolveToken(value);

        else
            decodedStr.push_back(static_cast<char>(value));
    }
    return decodedStr;
}
