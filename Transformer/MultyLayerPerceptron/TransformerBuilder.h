#pragma once

#include "tansformer/Encoder-Decoder-Transformer/Encoder-Decoder-Transformer.h"



class TransformerBuilder {

	private:
		//EncodeDecodeTransformer _transformer;

		size_t _embeddingSize;
		size_t _inputDictionarySize;
		size_t _outputDictionarySize;
		size_t _heads;

	public:
		TransformerBuilder();

		TransformerBuilder EmbeddingSize(size_t size);
		TransformerBuilder InputDictionarySize(size_t size);
		TransformerBuilder OutputDictionarySize(size_t size);
		TransformerBuilder Heads(size_t size);

		EncodeDecodeTransformer Build();

};
