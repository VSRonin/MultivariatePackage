#include "tDist.h"
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include "SpecialFunctions.hpp"
using namespace Multivariate;
double tDistribution::GetCumulativeDesity(const Eigen::VectorXd& Coordinates)const{
	//! \todo Add the Genz algorithm
	if(!AllValid || Coordinates.rows()!=Dim) return -1.0;
	Eigen::MatrixXd Samples=ExtractSamples(NumSimul);
	unsigned int Result=0;
	bool AllLess;
	for(unsigned int i=0;i<NumSimul;i++){
		AllLess=true;
		for(unsigned int j=0;j<Dim && AllLess;j++){
			if(Samples(i,j)>=Coordinates(j)) AllLess=false;
		}
		if(AllLess) Result++;
	}
	return static_cast<double>(Result)/static_cast<double>(NumSimul);
}
Eigen::MatrixXd tDistribution::ExtractSamples(unsigned int NumSamples) const{
	if(!AllValid || NumSamples==0) Eigen::MatrixXd();
	boost::random::uniform_real_distribution<double> dist(0.0, 1.0);
	NormalDistribution Numerator(Dim);
	Numerator.SetVarCovMatrix(VarCovMatrix);
	Numerator.SetRandomSeed(CurrentSeed);
	boost::math::chi_squared Denominator(DegreesOfFreedom);
	Eigen::MatrixXd Result=Numerator.ExtractSamples(NumSamples);
	for(unsigned int j=0;j<Dim;j++){
		for(unsigned int i=0;i<NumSamples;i++){
			Result(i,j)/=sqrt(quantile(Denominator,dist(RandNumGen))/static_cast<double>(DegreesOfFreedom));
			Result(i,j)+=meanVect(j);
		}
	}
	return Result;
}
double tDistribution::GetDensity(const Eigen::VectorXd& Coordinates)const{
	if(!AllValid) return -1.0;
	if(Coordinates.rows()!=Dim) return 0.0;
	double Result;
	if(Dim==1U){ //Univariate case
		Result= 
			(
				GammaFunction(static_cast<double>(DegreesOfFreedom+1U)/2.0) /
				(GammaFunction(static_cast<double>(DegreesOfFreedom)/2.0)*sqrt(VarCovMatrix(0,0)*boost::math::constants::pi<double>()*static_cast<double>(DegreesOfFreedom)))
			)*(
				pow(
					1.0+((1.0/(static_cast<double>(DegreesOfFreedom)*VarCovMatrix(0,0)))*(Coordinates(0)-meanVect(0))*(Coordinates(0)-meanVect(0)))
				,
					-static_cast<double>(DegreesOfFreedom+1U)/2.0
				)
			);
		return Result;
	}
	double distval=(Coordinates-meanVect).transpose()*VarCovMatrix.inverse()*(Coordinates-meanVect);
	Result=(
		GammaFunction(static_cast<double>(DegreesOfFreedom+Dim)/2.0) /
		(
			GammaFunction(static_cast<double>(DegreesOfFreedom)/2.0)
			*pow(static_cast<double>(DegreesOfFreedom),static_cast<double>(Dim)/2.0)
			*pow(boost::math::constants::pi<double>(),static_cast<double>(Dim)/2.0)
			*sqrt(VarCovMatrix.determinant())
			*pow(
				1.0+(distval/static_cast<double>(DegreesOfFreedom))
			,
				static_cast<double>(DegreesOfFreedom+Dim)/2.0
			)
		)
	);
	return Result;
}
tDistribution::tDistribution(unsigned int Dimension,unsigned int DegFreedom,const Eigen::VectorXd& locVect,const Eigen::MatrixXd& ScaleMatr)
	:NormalDistribution(Dimension,locVect,ScaleMatr)
	,DegreesOfFreedom(DegFreedom)
{
	AllValid= AllValid && DegreesOfFreedom>0;
}
tDistribution::tDistribution(unsigned int Dimension,unsigned int DegFreedom)
	:NormalDistribution(Dimension)
{
	AllValid= AllValid && DegreesOfFreedom>0;
	CurrentSeed=static_cast<unsigned int>(std::time(NULL));
	RandNumGen.seed(CurrentSeed);
}
tDistribution::tDistribution(const tDistribution& a)
	:DegreesOfFreedom(a.DegreesOfFreedom)
{
	Dim=a.Dim;
	AllValid=a.AllValid;
	ProbToFind=a.ProbToFind;
	meanVect=a.meanVect;
	VarCovMatrix=a.VarCovMatrix;
	CurrentSeed=a.CurrentSeed;
	RandNumGen.seed(CurrentSeed);
}
tDistribution& tDistribution::operator=(const tDistribution& a){
	DegreesOfFreedom=a.DegreesOfFreedom;
	Dim=a.Dim;
	meanVect=a.meanVect;
	VarCovMatrix=a.VarCovMatrix;
	AllValid=a.AllValid;
	ProbToFind=a.ProbToFind;
	return *this;
}
bool tDistribution::SetDegreesOfFreedom(unsigned int a){
	if(a>0){
		DegreesOfFreedom=a;
		CheckValidity();
		return true;
	}
	return false;
}
Eigen::MatrixXd tDistribution::ExtractSamplesCDF(unsigned int NumSamples) const{
	if(!AllValid || NumSamples<1U) return Eigen::MatrixXd();
	Eigen::MatrixXd Result=ExtractSamples(NumSamples);
	boost::math::students_t tDist(DegreesOfFreedom);
	for(unsigned int j=0;j<Dim;j++){
		for(unsigned int i=0;i<NumSamples;i++){
			Result(i,j)=cdf(tDist,(Result(i,j)/VarCovMatrix(j,j))-meanVect(j));
		}
	}
	return Result;
}
Eigen::MatrixXd tDistribution::GetVarMatrix() const{
	if(!AllValid || DegreesOfFreedom>2) return Eigen::MatrixXd();
	return (static_cast<double>(DegreesOfFreedom)/(static_cast<double>(DegreesOfFreedom)-2.0))*VarCovMatrix;
}