\brief Example usage code

This page contains examples of how to use the library in your code.

\tableofcontents
\section Normal Multivariate Normal
\subsection NormalEigen Using Eigen containers
\code
#include <iostream>
#include <vector>
#include "NormalDist.h"

int main(int argc, char* argv[])
{
	std::cout.precision(4); //Set the output to have 4 decimal digits

	//Construct the mean vector with values [1 ; -1]
	Eigen::Vector2d MeanVector;
	MeanVector(0)=1.0;
	MeanVector(1)=-1.0;

	//Construct the Var-Cov matrix so that the variance of both margins is 3 and the covariance is 1
	Eigen::Matrix2d VarMatrix;
	VarMatrix(0,0)=3;
	VarMatrix(0,1)=1;
	VarMatrix(1,0)=1;
	VarMatrix(1,1)=3;

	//Construct a Bivariate Normal with the above parameters
	Multivariate::NormalDistribution BivarNorm(2,MeanVector,VarMatrix);

	//Set the random number generator seed so that results are 100% reproducible
	BivarNorm.SetRandomSeed(99);

	//Compute and print the density and the cumulative density of the distribution in the point [1 ; 0.5]
	Eigen::Vector2d TempCoords;
	TempCoords(0)=1.0;
	TempCoords(1)=0.5;
	double Density=BivarNorm.GetDensity(TempCoords);
	double CumDensity=BivarNorm.GetCumulativeDesity(TempCoords);
	std::cout << "Density in [1 ; 0.5] = " << std::fixed << Density 
		<< std::endl << "Cumulative Density in [1 ; 0.5] = " << std::fixed << CumDensity ;

	//Simulate 10 realizations and print the results
	Eigen::Matrix<double,10,2> Samples=BivarNorm.ExtractSamples(10);
	std::cout << std::endl << "Simulation Results:" <<  std::endl << "Var 1\t Var2"<< std::endl << Samples << std::endl;

	return 0;
}
\endcode
The output produced will be:

<DFN>
Density in [1 ; 0.5] = 0.0369<br>
Cumulative Density in [1 ; 0.5] = 0.3778<br>
Simulation Results:<br>
Var 1	 Var2<br>
   1.1010   -3.4021<br>
   3.4808   -0.7299<br>
   0.4872   -2.3817<br>
  -1.2013   -0.2538<br>
   1.6614   -1.0759<br>
   2.4040   -3.2332<br>
   3.1768   -1.2504<br>
   0.0226   -3.3594<br>
   0.4933   -1.2479<br>
   2.7354   -0.1122<br>
</DFN>

\subsection NormalSTL Using STL containers
\code
#include <iostream>
#include <vector>
#include "NormalDist.h"

int main(int argc, char* argv[])
{
	std::cout.precision(4); //Set the output to have 4 decimal digits

	//Construct a Bivariate Normal Distribution
	Multivariate::NormalDistribution BivarNorm(2);

	//Set the random number generator seed so that results are 100% reproducible
	BivarNorm.SetRandomSeed(88);

	// Set the mean vector to [1 ; -1]
	std::vector<double> MeanVector(2);
	MeanVector[0]=1.0;
	MeanVector[1]=-1.0;
	BivarNorm.SetMeanVector(MeanVector);

	//Set the variance of both margins to 3 and the covariance to 1
	std::vector<double> VarMatrix(4);
	VarMatrix[0]=3.0;
	VarMatrix[1]=1.0;
	VarMatrix[2]=1.0;
	VarMatrix[3]=3.0;
	BivarNorm.SetVarCovMatrix(VarMatrix);

	//Compute and print the density and the cumulative density of the distribution in the point [1 ; 0.5]
	std::vector<double> TempCoords;
	TempCoords.push_back(1.0);
	TempCoords.push_back(0.5);
	double Density=BivarNorm.GetDensity(TempCoords);
	double CumDensity=BivarNorm.GetCumulativeDesity(TempCoords);
	std::cout << "Density in [1 ; 0.5] = " << std::fixed << Density 
		<< std::endl << "Cumulative Density in [1 ; 0.5] = " << std::fixed << CumDensity ;

	//Simulate 10 realizations and print the results
	std::map<unsigned int,std::vector<double> > Samples=BivarNorm.ExtractSamplesMap(10);
	std::cout << std::endl << "Simulation Results:"<< std::endl << "Var 1\t| Var2"<< std::endl << "_________________";
	for(int j=0;j<10;j++){
		std::cout << std::endl;
		for(unsigned int i=0;i<BivarNorm.GetDimension();i++){
			std::cout << std::fixed << Samples.at(i).at(j);
			 if(i<BivarNorm.GetDimension()-1) std::cout << "\t| ";
		}
	}
	std::cout << std::endl;

	return 0;
}
\endcode

The output produced will be:

<DFN>
Density in [1 ; 0.5] = 0.0369<br>
Cumulative Density in [1 ; 0.5] = 0.3773<br>
Simulation Results:<br>
Var 1	| Var2<br>
_________________<br>
2.0342	| -2.7569<br>
-0.8586	| -0.4041<br>
1.2382	| -2.8339<br>
1.9431	| -3.0771<br>
1.2090	| -1.9328<br>
4.0300	| -1.9376<br>
1.7155	| -1.8997<br>
-0.7756	| -2.4315<br>
-2.8239	| -1.5028<br>
-1.7615	| -1.6717<br>
</DFN>
