#pragma once

#include "GnuplotWrapper.h"



GnuplotWrap::GnuplotWrap()
{
	_gnuplotPipe  =  _popen("gnuplot -persist", "w"); 
}



GnuplotWrap::~GnuplotWrap()
{
	_pclose(_gnuplotPipe);

	if (out.is_open()) { out.close(); }
}



void GnuplotWrap::Plot(const std::string& str)
{
	std::string comand = "plot ";
	comand += str;

	fprintf(_gnuplotPipe, comand.c_str());
}

void GnuplotWrap::xRange(std::string xRangeStart, std::string xRangeEnd)
{
	std::string comand = "set xrange [" + xRangeStart + ":" + xRangeEnd +"] \n";
	fprintf(_gnuplotPipe, comand.c_str());
}


void GnuplotWrap::yRange(std::string yRangeStart, std::string yRangeEnd)
{
	std::string comand = "set yrange [" + yRangeStart + ":" + yRangeEnd +"] \n";
	fprintf(_gnuplotPipe, comand.c_str());
}


void GnuplotWrap::Grid(std::string x, std::string y)
{
	std::string comand = "set grid\nset ytics " + y + "\n" + "set xtics " + x + "\n";
	fprintf(_gnuplotPipe, comand.c_str());
}


void GnuplotWrap::OutFile(const std::string path)
{
	out = std::ofstream(path.c_str());
}



void GnuplotWrap::CloseFile()
{
	out.close();
}




GnuplotWrap& operator<<(GnuplotWrap& gnu, const std::string& str)
{
	fprintf(gnu._gnuplotPipe, str.c_str());
	return gnu;
}

GnuplotWrap& operator<<(GnuplotWrap& gnu, const int str)
{
	std::string value  =  std::to_string(str);
	fprintf(gnu._gnuplotPipe, value.c_str());
	return gnu;
}

GnuplotWrap& operator<<(GnuplotWrap& gnu, const float str)
{
	std::string value  =  std::to_string(str);
	fprintf(gnu._gnuplotPipe, value.c_str());
	return gnu;
}

GnuplotWrap& operator<<(GnuplotWrap& gnu, const double str)
{
	std::string value  =  std::to_string(str);
	fprintf(gnu._gnuplotPipe, value.c_str());
	return gnu;
}
