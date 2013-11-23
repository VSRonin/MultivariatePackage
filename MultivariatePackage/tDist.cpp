#include "tDist.h"
#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/distributions/gamma.hpp>
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
	Numerator.SetVarCovMatrix(ScaleMatrix);
	Numerator.SetRandomSeed(CurrentSeed);
	boost::math::chi_squared Denominator(DegreesOfFreedom);
	Eigen::MatrixXd Result=Numerator.ExtractSamples(NumSamples);
	for(unsigned int j=0;j<Dim;j++){
		for(unsigned int i=0;i<NumSamples;i++){
			Result(i,j)/=sqrt(quantile(Denominator,dist(RandNumGen))/static_cast<double>(DegreesOfFreedom));
			Result(i,j)+=LocatVect(j);
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
				boost::math::tgamma(static_cast<double>(DegreesOfFreedom+1U)/2.0) /
				(boost::math::tgamma(static_cast<double>(DegreesOfFreedom)/2.0)*sqrt(ScaleMatrix(0,0)*boost::math::constants::pi<double>()*static_cast<double>(DegreesOfFreedom)))
			)*(
				pow(
					1.0+((1.0/(static_cast<double>(DegreesOfFreedom)*ScaleMatrix(0,0)))*(Coordinates(0)-LocatVect(0))*(Coordinates(0)-LocatVect(0)))
				,
					-static_cast<double>(DegreesOfFreedom+1U)/2.0
				)
			);
		return Result;
	}
	double distval=(Coordinates-LocatVect).transpose()*ScaleMatrix.inverse()*(Coordinates-LocatVect);
	Result=(
		boost::math::tgamma(static_cast<double>(DegreesOfFreedom+Dim)/2.0) /
		(
			boost::math::tgamma(static_cast<double>(DegreesOfFreedom)/2.0)
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
	return Result;
}
tDistribution::tDistribution(unsigned int Dimension,unsigned int DegFreedom,const Eigen::VectorXd& locVect,const Eigen::MatrixXd& ScaleMatr)
	:AbstarctDistribution(Dimension)
	,LocatVect(locVect)
	,ScaleMatrix(ScaleMatr)
	,DegreesOfFreedom(DegFreedom)
	,UseGenz(true)
	,NumSimul(500000U)
{
	CheckValidity();
}
tDistribution::tDistribution(unsigned int Dimension,unsigned int DegFreedom)
	:AbstarctDistribution(Dimension)
	,DegreesOfFreedom(DegFreedom)
	,UseGenz(true)
	,NumSimul(500000U)
{
	if(Dimension>0){
		for(unsigned int i=0;i<Dim;i++){
			LocatVect(i)=0.0;
			for(unsigned int j=0;j<Dim;j++){
				if(i==j) ScaleMatrix(i,j)=1.0;
				else ScaleMatrix(i,j)=0.0;
			}
		}
	}
	CheckValidity();
}
tDistribution::tDistribution(const tDistribution& a)
	:DegreesOfFreedom(a.DegreesOfFreedom)
	,ProbToFind(a.ProbToFind)
	,LocatVect(a.LocatVect)
	,ScaleMatrix(a.ScaleMatrix)
	,NumSimul(a.NumSimul)
	,UseGenz(a.UseGenz)
{
	Dim=a.Dim;
	AllValid=a.AllValid;
	CurrentSeed=a.CurrentSeed;
	RandNumGen.seed(CurrentSeed);
}
tDistribution& tDistribution::operator=(const tDistribution& a){
	DegreesOfFreedom=a.DegreesOfFreedom;
	Dim=a.Dim;
	LocatVect=a.LocatVect;
	ScaleMatrix=a.ScaleMatrix;
	AllValid=a.AllValid;
	ProbToFind=a.ProbToFind;
	NumSimul=a.NumSimul;
	UseGenz=a.UseGenz;
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
			Result(i,j)=cdf(tDist,(Result(i,j)/ScaleMatrix(j,j))-LocatVect(j));
		}
	}
	return Result;
}
Eigen::MatrixXd tDistribution::GetVarMatrix() const{
	if(!AllValid || DegreesOfFreedom>2) return Eigen::MatrixXd();
	return (static_cast<double>(DegreesOfFreedom)/(static_cast<double>(DegreesOfFreedom)-2.0))*ScaleMatrix;
}
bool tDistribution::CheckValidity(){
	AllValid=true;
	if(Dim<1U){ //The dimension must be at least one
		AllValid=false;
		return AllValid;
	}
	if(DegreesOfFreedom<1U){ //The degrees of freedom must be at least one
		AllValid=false;
		return AllValid;
	}
	if(LocatVect.rows()!=Dim){ //The mean vector must be as many elements as there are dimensions
		AllValid=false;
		return AllValid;
	}
	if(ScaleMatrix.rows()!=ScaleMatrix.cols() || ScaleMatrix.rows()!=Dim){ //The Var-Cov Matrix must be squared and have as many rows as there are dimensions
		AllValid=false;
		return AllValid;
	}
	if(ScaleMatrix!=ScaleMatrix.transpose()){ //The Var-Cov Matrix must be symmetric
		AllValid=false;
		return AllValid;
	}
	//The Variance Covariance matrix must be positive definite
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
bool tDistribution::SetLocationVector(const std::vector<double>& mVect){
	if(mVect.size()!=Dim) return false;
	for(unsigned int i=0;i<Dim;i++){
		LocatVect(i)=mVect.at(i);
	}
	CheckValidity();
	return true;
}
bool tDistribution::SetDimension(unsigned int Dimension){
	if(Dimension==0U) return false;
	Dim=Dimension;
	for(unsigned int i=0;i<Dim;i++){
		LocatVect(i)=0.0;
		for(unsigned int j=0;j<Dim;j++){
			if(i==j) ScaleMatrix(i,j)=1.0;
			else ScaleMatrix(i,j)=0.0;
		}
	}
	CheckValidity();
	return true;
}
bool tDistribution::SetScaleMatrix(const Eigen::MatrixXd& SclMatr){
	if(SclMatr.rows()!=SclMatr.cols() || SclMatr.rows()!=Dim) //The Var-Cov Matrix must be squared and have as many rows as there are dimensions
		return false;
	if(SclMatr!=SclMatr.transpose()) //The Var-Cov Matrix must be symmetric
		return false;
	//The Variance Covariance matrix must be positive definite
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
Eigen::VectorXd tDistribution::GetQuantile(double Prob)const{
	if(!AllValid || Prob>1.0 || Prob<0.0) return Eigen::VectorXd();
	if(Prob==1.0 || Prob==0.0){
		Eigen::VectorXd TempVector(Dim);
		for(unsigned int i=0;i<Dim;i++){
			TempVector(i)= Prob>0.0 ? DBL_MAX : -DBL_MAX;
		}
		return TempVector;
	}
	tDistribution CentralDistr(Dim);
	CentralDistr.SetScaleMatrix(ScaleMatrix);
	CentralDistr.SetRandomSeed(CurrentSeed);
	CentralDistr.ProbToFind=Prob;
	double CenteredQuantile =  boost::math::tools::newton_raphson_iterate(CentralDistr,0.0,-DBL_MAX,DBL_MAX,8);
	Eigen::VectorXd CoordinatesVector(Dim);
	for(unsigned i=0;i<Dim;i++){
		if(LocatVect(i)>0.0){
			if(CenteredQuantile>DBL_MAX-LocatVect(i)) CoordinatesVector(i)=DBL_MAX;
			else CoordinatesVector(i)= CenteredQuantile+LocatVect(i);
		}
		else {
			if(CenteredQuantile<-DBL_MAX-LocatVect(i)) CoordinatesVector(i)=-DBL_MAX;
			else CoordinatesVector(i)= CenteredQuantile+LocatVect(i);
		}
	}
	return CoordinatesVector;
}
boost::math::tuple<double, double> tDistribution::operator()(double x){
	Eigen::VectorXd CoordinatesVector(Dim);
	for(unsigned i=0;i<Dim;i++) CoordinatesVector(i)=x;
	return boost::math::make_tuple(GetCumulativeDesity(CoordinatesVector)-ProbToFind,GetDensity(CoordinatesVector));
}
std::pair<double,double> tDistribution::ComputeThat(unsigned int N)const{
	Eigen::MatrixXd CholVar = Eigen::LLT<Eigen::MatrixXd>(ScaleMatrix).matrixL(); // compute the Cholesky decomposition of the Scale matrix
	boost::random::uniform_real_distribution<double> UnifDist(0.0,1.0);
	boost::math::normal StandardNormal(0.0,1.0);

	boost::math::gamma_distribution<double> GammaDist(1.0/Theta,1.0);
	
}