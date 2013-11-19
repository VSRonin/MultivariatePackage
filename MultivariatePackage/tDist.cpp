#include "tDist.h"
#include "NormalDist.h"
#include <ctime>
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include "SpecialFunctions.hpp"
using namespace Multivariate;
double tDistribution::GetCumulativeDesity(const Eigen::VectorXd& Coordinates, bool UseGenz, unsigned int NumSimul)const{
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
	Numerator.SetVarCovMatrix(ScaleMatrix);
	Numerator.SetRandomSeed(CurrentSeed);
	boost::math::chi_squared Denominator(DegreesOfFreedom);
	Eigen::MatrixXd Result=Numerator.ExtractSamples(NumSamples);
	for(unsigned int j=0;j<Dim;j++){
		for(unsigned int i=0;i<NumSamples;i++){
			Result(i,j)/=sqrt(quantile(Denominator,dist(RandNumGen))/static_cast<double>(DegreesOfFreedom));
			Result(i,j)+=LocationVect(j);
		}
	}
	return Result;
}
double tDistribution::GetDensity(const Eigen::VectorXd& Coordinates, bool GetLogDensity)const{
	if(!AllValid) return -1.0;
	if(Coordinates.rows()!=Dim) return 0.0;
	double Result;
	if(Dim==1U){ //Univariate case
		Result= 
			(
				GammaFunction(static_cast<double>(DegreesOfFreedom+1U)/2.0) /
				(GammaFunction(static_cast<double>(DegreesOfFreedom)/2.0)*sqrt(ScaleMatrix(0,0)*boost::math::constants::pi<double>()*static_cast<double>(DegreesOfFreedom)))
			)*(
				pow(
					1.0+((1.0/(static_cast<double>(DegreesOfFreedom)*ScaleMatrix(0,0)))*(Coordinates(0)-LocationVect(0))*(Coordinates(0)-LocationVect(0)))
				,
					-static_cast<double>(DegreesOfFreedom+1U)/2.0
				)
			);
		if(GetLogDensity) return log(Result);
		else return Result;
	}
	double distval=(Coordinates-LocationVect).transpose()*ScaleMatrix.inverse()*(Coordinates-LocationVect);
	Result=(
		GammaFunction(static_cast<double>(DegreesOfFreedom+Dim)/2.0) /
		(
			GammaFunction(static_cast<double>(DegreesOfFreedom)/2.0)
			*pow(static_cast<double>(DegreesOfFreedom),static_cast<double>(Dim)/2.0)
			*pow(boost::math::constants::pi<double>(),static_cast<double>(Dim)/2.0)
			*sqrt(ScaleMatrix.determinant())
			*pow(
				1.0+(distval/static_cast<double>(DegreesOfFreedom))
			,
				static_cast<double>(DegreesOfFreedom+Dim)/2.0
			)
		)
	);
	if(GetLogDensity) return log(Result);
	else return Result;
}
tDistribution::tDistribution(unsigned int Dimension,unsigned int DegFreedom,const Eigen::VectorXd& locVect,const Eigen::MatrixXd& ScaleMatr)
	:Dim(Dimension)
	,DegreesOfFreedom(DegFreedom)
	,LocationVect(locVect)
	,ScaleMatrix(ScaleMatr)
	,ProbToFind(0.0)
{
	CheckValidity();
	CurrentSeed=static_cast<unsigned int>(std::time(NULL));
	RandNumGen.seed(CurrentSeed);
}
tDistribution::tDistribution(unsigned int Dimension,unsigned int DegFreedom)
	:Dim(Dimension)
	,DegreesOfFreedom(DegFreedom)
	,LocationVect(Eigen::VectorXd(Dimension))
	,ScaleMatrix(Eigen::MatrixXd(Dimension,Dimension))
	,ProbToFind(0.0)
{
	if(Dimension>0U){
		for(unsigned int i=0;i<Dim;i++){
			LocationVect(i)=0.0;
			for(unsigned int j=0;j<Dim;j++){
				if(i==j) ScaleMatrix(i,j)=1.0;
				else ScaleMatrix(i,j)=0.0;
			}
		}
	}
	CheckValidity();
	CurrentSeed=static_cast<unsigned int>(std::time(NULL));
	RandNumGen.seed(CurrentSeed);
}
tDistribution::tDistribution(const tDistribution& a)
	:DegreesOfFreedom(a.DegreesOfFreedom)
	,Dim(a.Dim)
	,LocationVect(a.LocationVect)
	,ScaleMatrix(a.ScaleMatrix)
	,AllValid(a.AllValid)
	,ProbToFind(a.ProbToFind)
{
	CurrentSeed=a.CurrentSeed;
	RandNumGen.seed(CurrentSeed);
}
tDistribution& tDistribution::operator=(const tDistribution& a){
	DegreesOfFreedom=a.DegreesOfFreedom;
	Dim=a.Dim;
	LocationVect=a.LocationVect;
	ScaleMatrix=a.ScaleMatrix;
	AllValid=a.AllValid;
	ProbToFind=a.ProbToFind;
	return *this;
}
void tDistribution::SetRandomSeed(unsigned int NewSeed){
	CurrentSeed=NewSeed;
	RandNumGen.seed(NewSeed);
}
bool tDistribution::CheckValidity(){
	AllValid=true;
	if(Dim<1U || DegreesOfFreedom<1U){ //The dimension and degrees of freedom must be at least one
		AllValid=false;
		return AllValid;
	}
	if(LocationVect.rows()!=Dim){ //The mean vector must be as many elements as there are dimensions
		AllValid=false;
		return AllValid;
	}
	if(ScaleMatrix.rows()!=ScaleMatrix.cols() || ScaleMatrix.rows()!=Dim){ //The Scale Matrix must be squared and have as many rows as there are dimensions
		AllValid=false;
		return AllValid;
	}
	if(ScaleMatrix!=ScaleMatrix.transpose()){ //The Scale Matrix must be symmetric
		AllValid=false;
		return AllValid;
	}
	//The Scale matrix must be positive definite
	if(ScaleMatrix.determinant()<=0.0){
		AllValid=false;
		return AllValid;
	}
	Eigen::VectorXcd RelatedEigen=ScaleMatrix.eigenvalues();
	for (unsigned int i=0;i<Dim;i++){
		if(RelatedEigen(i).real()<0.0){ 
			AllValid=false;
			return AllValid;
		}
	}
	return AllValid;
}
bool tDistribution::SetLocationVector(const Eigen::VectorXd& mVect){
	if(mVect.rows()!=Dim) return false;
	LocationVect=mVect;
	CheckValidity();
	return true;
}
bool tDistribution::SetLocationVector(const std::vector<double>& mVect){
	if(mVect.size()!=Dim) return false;
	Eigen::VectorXd TempVector(Dim);
	for (unsigned int i=0;i<Dim;i++) TempVector(i)=mVect[i];
	return SetLocationVector(TempVector);
}
bool tDistribution::SetDegreesOfFreedom(unsigned int a){
	if(a>0){
		DegreesOfFreedom=a;
		return true;
	}
	return false;
}
bool tDistribution::SetDimension(unsigned int Dimension){
	if(Dimension==0U) return false;
	Dim=Dimension;
	for(unsigned int i=0;i<Dim;i++){
		LocationVect(i)=0.0;
		for(unsigned int j=0;j<Dim;j++){
			if(i==j) ScaleMatrix(i,j)=1.0;
			else ScaleMatrix(i,j)=0.0;
		}
	}
	CheckValidity();
	return true;
}
bool tDistribution::SetScaleMatrix(const Eigen::MatrixXd& SclMatr){
	if(SclMatr.rows()!=SclMatr.cols() || SclMatr.rows()!=Dim) //The Scale Matrix must be squared and have as many rows as there are dimensions
		return false;
	if(SclMatr!=SclMatr.transpose()) //The Scale Matrix must be symmetric
		return false;
	//The Scale matrix must be positive definite
	if(SclMatr.determinant()<0.0)
		return false;

	Eigen::VectorXcd RelatedEigen=SclMatr.eigenvalues();
	for (unsigned int i=0;i<Dim;i++){
		if(RelatedEigen(i).real()<0.0){ 
			return false;
		}
	}
	ScaleMatrix=SclMatr;
	CheckValidity();
	return true;
}
bool tDistribution::SetScaleMatrix(const std::vector<double>& mVect, bool RowWise){
	if(mVect.size()!=Dim*Dim) return false;
	Eigen::MatrixXd TempMatrix(Dim,Dim);
	for(unsigned int i=0;i<mVect.size();i++){
		TempMatrix(i/Dim,i%Dim)=mVect.at(i);
	}
	if(!RowWise) TempMatrix.transposeInPlace();
	return SetScaleMatrix(TempMatrix);
}
std::vector<double> tDistribution::ExtractSampleVector() const{
	std::vector<double> Result(Dim);
	if(!AllValid) return Result;
	Eigen::RowVectorXd TempVector=ExtractSample();
	for(unsigned int i=0;i<Dim;i++){
		Result[i]=TempVector(i);
	}
	return Result;
}
std::map<unsigned int,std::vector<double> > tDistribution::ExtractSamplesMap(unsigned int NumSamples) const{
	if(!AllValid || NumSamples==0) return std::map<unsigned int,std::vector<double> >();
	std::map<unsigned int,std::vector<double> > Result;
	std::vector<double> Series(NumSamples);
	Eigen::MatrixXd TempMatrix=ExtractSamples(NumSamples);
	for(unsigned int i=0;i<Dim;i++){
		for(unsigned int j=0;j<NumSamples;j++){
			Series[j]=TempMatrix(j,i);
		}
		Result.insert(std::pair<unsigned int,std::vector<double> >(i,Series));
	}
	return Result;
}
double tDistribution::GetDensity(const std::vector<double>& Coordinates, bool GetLogDensity)const{
	if(Coordinates.size()!=Dim) return 0.0;
	Eigen::VectorXd TempVector(Dim);
	for(unsigned int i=0;i<Dim;i++){
		TempVector(i)=Coordinates.at(i);
	}
	return GetDensity(TempVector,GetLogDensity);
}
double tDistribution::GetCumulativeDesity(const std::vector<double>& Coordinates, bool UseGenz, unsigned int NumSimul)const{
	if(Coordinates.size()!=Dim) return 0.0;
	Eigen::VectorXd TempVector(Dim);
	for(unsigned int i=0;i<Dim;i++){
		TempVector(i)=Coordinates.at(i);
	}
	return GetCumulativeDesity(TempVector,UseGenz,NumSimul);
}
Eigen::VectorXd tDistribution::GetQuantile(double Prob)const{
	if(!AllValid || Prob>1.0 || Prob<0.0) return Eigen::VectorXd();
	if(Prob==1.0 || Prob==0.0){
		Eigen::VectorXd TempVector(Dim);
		for(unsigned int i=0;i<Dim;i++){
			TempVector(i)= Prob>0.0 ? DBL_MAX : -DBL_MAX;
		}
		return TempVector;
	}
	tDistribution CentralDistr(Dim,DegreesOfFreedom);
	CentralDistr.SetScaleMatrix(ScaleMatrix);
	CentralDistr.SetRandomSeed(CurrentSeed);
	CentralDistr.ProbToFind=Prob;
	double CenteredQuantile =  boost::math::tools::newton_raphson_iterate(CentralDistr,0.0,-DBL_MAX,DBL_MAX,8);
	Eigen::VectorXd CoordinatesVector(Dim);
	for(unsigned i=0;i<Dim;i++){
		if(LocationVect(i)>0.0){
			if(CenteredQuantile>DBL_MAX-LocationVect(i)) CoordinatesVector(i)=DBL_MAX;
			else CoordinatesVector(i)= CenteredQuantile+LocationVect(i);
		}
		else {
			if(CenteredQuantile<-DBL_MAX-LocationVect(i)) CoordinatesVector(i)=-DBL_MAX;
			else CoordinatesVector(i)= CenteredQuantile+LocationVect(i);
		}
	}
	return CoordinatesVector;
}
std::vector<double> tDistribution::GetQuantileVector(double Prob)const{
	if(!AllValid || Prob>1.0 || Prob<0.0) return std::vector<double>();
	Eigen::VectorXd TempVector=GetQuantile(Prob);
	std::vector<double> Result(Dim);
	for(unsigned int i=0;i<Dim;i++) Result[i]=TempVector(i);
	return Result;
}
boost::math::tuple<double, double> tDistribution::operator()(double x){
	Eigen::VectorXd CoordinatesVector(Dim);
	for(unsigned i=0;i<Dim;i++) CoordinatesVector(i)=x;
	return boost::math::make_tuple(GetCumulativeDesity(CoordinatesVector)-ProbToFind,GetDensity(CoordinatesVector));
}