#include "UniformDistribution.h"
#include <boost/random/uniform_real_distribution.hpp>
#include <ctime>
using namespace Multivariate;
UniformDistribution::UniformDistribution(unsigned int Dimension)
	:AbstarctDistribution(Dimension)
{
	AllValid=Dim>0;
	if(AllValid){
		Eigen::MatrixX2d TempLimits(Dimension,2);
		for(unsigned int i=0;i<Dim;i++){
			TempLimits(i,0)=0.0;
			TempLimits(i,1)=1.0;
		}
		Limits=TempLimits;
	}
}
UniformDistribution::UniformDistribution(unsigned int Dimension, const Eigen::MatrixX2d& MinMax)
	:AbstarctDistribution(Dimension)
	,Limits(MinMax)
{
	AllValid=Dim>0 && MinMax.rows()==Dim;
	double TempValue;
	for(unsigned int i=0;i<Dim && AllValid;i++){
		if(Limits(i,0)==Limits(i,1)) AllValid=false;
		else if(Limits(i,0)>Limits(i,1)){
			TempValue=Limits(i,0);
			Limits(i,0)=Limits(i,1);
			Limits(i,1)=TempValue;
		}
	}
}
Eigen::MatrixXd UniformDistribution::ExtractSamples(unsigned int NumSamples) const{
	Eigen::MatrixXd Result(NumSamples,Dim);
	for(unsigned int j=0;j<Dim;j++){
		boost::random::uniform_real_distribution<double> dist(Limits(j,0), Limits(j,1));
		for(unsigned int i=0;i<NumSamples;i++){
			Result(i,j)=dist(RandNumGen);
		}
	}
	return Result;
}
double UniformDistribution::GetDensity()const{
	if(!AllValid) return -1.0;
	double Result=1.0;
	for(unsigned int j=0;j<Dim;j++){
		Result*=Limits(j,1)-Limits(j,0);
	}
	return 1.0/Result;
}
double UniformDistribution::GetCumulativeDesity(const Eigen::VectorXd& Coordinates)const{
	if(!AllValid || Coordinates.rows()!=Dim) return -1.0;
	double Result=1.0;
	for(unsigned int i=0;i<Dim;i++){
		if(Coordinates(i)<Limits(i,0) || Coordinates(i)>Limits(i,1)) return -1.0;
		Result*=Coordinates(i)-Limits(i,0);
	}
	return Result*GetDensity();
}
Eigen::VectorXd UniformDistribution::GetQuantile(double Prob)const{
	if(Prob>1.0 || Prob<0.0 || !AllValid) return Eigen::VectorXd();
	double Equidistant;
	if(Prob==1.0) Equidistant=1.0;
	else if(Prob==0.0) Equidistant=0.0;
	else Equidistant= pow(Prob/GetDensity(),1.0/static_cast<double>(Dim));
	Eigen::VectorXd Result(Dim);
	for(unsigned int i=0;i<Dim;i++) Result(i)=Limits(i,0)+Equidistant;
	return Result;
}

bool UniformDistribution::SetDimension(unsigned int Dimension){
	if(Dimension<1U) return false;
	Dim=Dimension;
	Eigen::Matrix<double,-1,2> TempLimits(Dimension,2);
	for(unsigned int i=0;i<Dim;i++){
		TempLimits(i,0)=0.0;
		TempLimits(i,1)=1.0;
	}
	Limits=TempLimits;
	AllValid=true;
	return true;
}
bool UniformDistribution::SetLimits(const Eigen::MatrixX2d& MinMax){
	if(MinMax.rows()!=Dim) return false;
	Eigen::MatrixX2d TempLimits(Dim,2);
	double TempValue;
	for(unsigned int i=0;i<Dim && AllValid;i++){
		if(TempLimits(i,0)==TempLimits(i,1)) return false;
		else if(TempLimits(i,0)>TempLimits(i,1)){
			TempValue=TempLimits(i,0);
			TempLimits(i,0)=TempLimits(i,1);
			TempLimits(i,1)=TempValue;
		}
	}
	Limits=TempLimits;
	AllValid=true;
	return true;
}
Eigen::MatrixXd UniformDistribution::ExtractSamplesCDF(unsigned int NumSamples)const{
	Eigen::MatrixXd Result(NumSamples,Dim);
	boost::random::uniform_real_distribution<double> dist(0.0,1.0);
	for(unsigned int i=0;i<NumSamples;i++){
		for(unsigned int j=0;j<Dim;j++){
			Result(i,j)=dist(RandNumGen);
		}
	}
	return Result;
}