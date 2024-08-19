#pragma once

#include "../utils/basic-includes.h"



class IActivationFunction {
	public:
	virtual double f(double x) = 0;
	virtual double df(double x) = 0;
	virtual const char* ToString() = 0;
};





class CustonActivationFunction : public IActivationFunction {
	private:
	std::function<double(double x)> _function;
	double _h;

	public:
	CustonActivationFunction(std::function<double(double x)> f, double h =  1.e-11);
	~CustonActivationFunction();

	virtual double f(double x) override;
	virtual double df(double x) override;
	virtual const char* ToString() override;
};




class Sigmoid : public IActivationFunction {
	private:
		double _a;
	public:
		Sigmoid(double a = 1.00);
		virtual double f(double x) override;
		virtual double df(double x) override;
		virtual const char* ToString() override;
};



class AdaptedSigmoid : public IActivationFunction {
	public:
		virtual double f(double x) override;
		virtual double df(double x) override;
		virtual const char* ToString() override;
};



class ReLU : public IActivationFunction {
	public:
		virtual double f(double x) override;
		virtual double df(double x) override;
		virtual const char* ToString() override;
};



class LeakyReLU : public IActivationFunction {
	private:
		double _a;
	public:
		LeakyReLU(double a = 0.1);
		virtual double f(double x) override;
		virtual double df(double x) override;
		virtual const char* ToString() override;
};



class Tanh : public IActivationFunction {
	public:
		virtual double f(double x) override;
		virtual double df(double x) override;
		virtual const char* ToString() override;
};



class NormalizedTanh : public IActivationFunction {
	public:
		virtual double f(double x) override;
		virtual double df(double x) override;
		virtual const char* ToString() override;
};



class Linear : public IActivationFunction {
	public:
		virtual double f(double x) override;
		virtual double df(double x) override;
		virtual const char* ToString() override;
};



class ClipedLinear : public IActivationFunction {
	private:
		double _min;
		double _max;
	public:
		ClipedLinear(double min, double max);

		virtual double f(double x) override;
		virtual double df(double x) override;
		virtual const char* ToString() override;
};


