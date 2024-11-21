#pragma once

#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>
#include <limits>
#include <cmath>
#include <numeric>
#include <string>


class GnuplotWrap {

	private:
		FILE* _gnuplotPipe;

	public:
		std::ofstream out;

		GnuplotWrap();
		~GnuplotWrap();

		void Plot(const std::string& str);
		void xRange(std::string xRangeStart, std::string xRangeEnd);
		void yRange(std::string yRangeStart, std::string yRangeEnd);
		void OutFile(const std::string path);
		void Grid(std::string x, std::string y);
		void CloseFile();


		friend GnuplotWrap& operator<<(GnuplotWrap& gnu, const std::string& str);
		friend GnuplotWrap& operator<<(GnuplotWrap& gnu, const int str);
		friend GnuplotWrap& operator<<(GnuplotWrap& gnu, const float str);
		friend GnuplotWrap& operator<<(GnuplotWrap& gnu, const double str);

};
