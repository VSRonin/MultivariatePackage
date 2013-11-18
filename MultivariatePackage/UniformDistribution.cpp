#include "UniformDistribution.h"
#include <boost/random/uniform_real_distribution.hpp>
#include <ctime>
using namespace Multivariate;
UniformDistribution::UniformDistribution(unsigned int Dimension)
	:Dim(Dimension)
{
	AllValid=Dim>0;
	if(AllValid){
		Eigen::Matrix<double,-1,2> TempLimits(Dimension);
		for(unsigned int i=0;i<Dim;i++){
			TempLimits(i,0)=0.0;
			TempLimits(i,1)=1.0;
		}
		Limits=TempLimits;
	}
	CurrentSeed=static_cast<unsigned int>(std::time(NULL));
	RandNumGen.seed(CurrentSeed);
}
UniformDistribution::UniformDistribution(unsigned int Dimension, const Eigen::Matrix<double,-1,2>& MinMax)
	:Dim(Dimension)
	,Limits(MinMax)
{
	AllValid=Dim>0 && MinMax.rows()==Dim;
	double TempValue;
	for(unsigned int i=0;i<Dim && AllValid;i++){
		if(Limits(i,0)>Limits(i,1)){
			TempValue=Limits(i,0);
			Limits(i,0)=Limits(i,1);
			Limits(i,1)=TempValue;
		}
	}
	CurrentSeed=static_cast<unsigned int>(std::time(NULL));
	RandNumGen.seed(CurrentSeed);
}
void UniformDistribution::SetRandomSeed(unsigned int NewSeed){
	CurrentSeed=NewSeed;
	RandNumGen.seed(CurrentSeed);
}
Eigen::MatrixXd UniformDistribution::ExtractSamples(unsigned int NumSamples) const{
	Eigen::MatrixXd Result(NumSamples,Dim);
	for(unsigned int j=0;j<Dim;j++){
		if(Limits(j,0)==Limits(j,1)){
			for(unsigned int i=0;i<NumSamples;i++) Result(i,j)=Limits(j,0);
		}
		else{
			boost::random::uniform_real_distribution<double> dist(Limits(j,0), Limits(j,1));
			for(unsigned int i=0;i<NumSamples;i++){
				Result(i,j)=dist(RandNumGen);
			}
		}
	}
	return Result;
}
double UniformDistribution::GetDensity(bool GetLogDensity=false)const{
	double Result=1.0;
	for(unsigned int j=0;j<Dim;j++){
		if(Limits(j,0)!=Limits(j,1)) Result*=Limits(j,1)-Limits(j,0);
	}
	if(GetLogDensity) log(1.0/Result);
	else return 1.0/Result;
}
double UniformDistribution::GetCumulativeDesity(const Eigen::VectorXd& Coordinates)const{
	return 0.0; //TODO
}