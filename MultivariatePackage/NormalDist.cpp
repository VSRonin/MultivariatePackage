#include "NormalDist.h"
#include <Eigen/Eigenvalues>
#include <Eigen/Cholesky>
#include <boost/math/distributions/normal.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/math/constants/constants.hpp>
#include <ctime>
using namespace Multivariate;
Eigen::VectorXd NormalDistribution::GetQuantile(double Prob)const{
	if(!AllValid || Prob>1.0 || Prob<0.0) return Eigen::VectorXd();
	if(Prob==1.0 || Prob==0.0){
		Eigen::VectorXd TempVector(Dim);
		for(unsigned int i=0;i<Dim;i++){
			TempVector(i)= Prob>0.0 ? sqrt(DBL_MAX) : -sqrt(DBL_MAX);
		}
		return TempVector;
	}
	NormalDistribution CentralDistr(Dim);
	CentralDistr.SetVarCovMatrix(VarCovMatrix);
	CentralDistr.SetRandomSeed(CurrentSeed);
	CentralDistr.ProbToFind=Prob;
	double CenteredQuantile =  boost::math::tools::newton_raphson_iterate(CentralDistr,0.0,-sqrt(DBL_MAX),sqrt(DBL_MAX),8);
	Eigen::VectorXd CoordinatesVector(Dim);
	for(unsigned i=0;i<Dim;i++){
		if(meanVect(i)>0.0){
			if(CenteredQuantile>DBL_MAX-meanVect(i)) CoordinatesVector(i)=DBL_MAX;
			else CoordinatesVector(i)= CenteredQuantile+meanVect(i);
		}
		else {
			if(CenteredQuantile<-DBL_MAX-meanVect(i)) CoordinatesVector(i)=-DBL_MAX;
			else CoordinatesVector(i)= CenteredQuantile+meanVect(i);
		}
	}
	return CoordinatesVector;
}
double NormalDistribution::GetCumulativeDesity(const Eigen::VectorXd& Coordinates)const{
	if(!AllValid || Coordinates.rows()!=Dim || NumSimul<1U) return -1.0;
	if(Dim==1U){ //Univariate Case
		boost::math::normal NormalDist(meanVect(0),VarCovMatrix(0,0));
		return boost::math::cdf(NormalDist,Coordinates(0));
	}
	if(UseGenz){
		//Sort the variables based on the coordinates to speed up the calculations
		std::vector<unsigned int> VariablesOrder=FillOrder(Coordinates);
		Eigen::VectorXd AdjCoordinates(Dim);
		Eigen::VectorXd AdjMean(Dim);
		Eigen::MatrixXd AdjVarCoV(Dim,Dim);
		for(unsigned int i=0;i<Dim;i++){
			AdjCoordinates(VariablesOrder[i])=Coordinates(i);
			AdjMean(VariablesOrder[i])=meanVect(i);
			for(unsigned int j=0;j<Dim;j++){
				AdjVarCoV(VariablesOrder[i],VariablesOrder[j])=VarCovMatrix(i,j);
			}
		}
		//Run the monte-carlo simulation
		const double MonteCarloConfidenceFactor=2.5;
		boost::random::uniform_real_distribution<double> dist(0.0, 1.0);
		boost::math::normal StandardNormal(0.0,1.0);
		Eigen::MatrixXd CholVar = Eigen::LLT<Eigen::MatrixXd>(AdjVarCoV).matrixL(); // compute the Cholesky decomposition of the Var-Cov matrix
		double IntSum=0.0, VarSum=0.0;
		double SumCy, delta;
		unsigned int Counter=0;
		//The parameter d is always 0.0 since the lower bound is -Infinity
		double *e=new double[Dim]; *e=boost::math::cdf(StandardNormal,(AdjCoordinates(0)-AdjMean(0))/CholVar(0,0));
		double *f=new double[Dim]; *f=*e;
		double *y=new double[Dim-1];
		do{
			for(unsigned int i=1;i<Dim;i++){
				if(e[i-1]>0.0 && e[i-1]<1.0) y[i-1]=boost::math::quantile(StandardNormal,dist(RandNumGen)*(e[i-1]));
				else if(e[i-1]>0.0) y[i-1]=sqrt(DBL_MAX);
				else if(e[i-1]<1.0)	y[i-1]=-sqrt(DBL_MAX);
				SumCy=0.0;
				for(unsigned int j=0;j<i;j++){
					SumCy+=CholVar(i,j)*(y[j]);
				}
				e[i]=boost::math::cdf(StandardNormal,((AdjCoordinates(i)-AdjMean(i))-SumCy)/CholVar(i,i));
				f[i]=e[i]*(f[i-1]);
			}
			Counter++;
			delta=(f[Dim-1]-IntSum)/static_cast<double>(Counter);
			IntSum+=delta;
			VarSum=(static_cast<double>(Counter-2U)*VarSum/static_cast<double>(Counter))+(delta*delta);
		}while(MonteCarloConfidenceFactor*sqrt(VarSum)>=0.005 && Counter<NumSimul);
		delete [] e;
		delete [] f;
		delete [] y;
		return IntSum;
	}
	else {
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
}
Eigen::MatrixXd NormalDistribution::ExtractSamples(unsigned int NumSamples) const{
	if(!AllValid || NumSamples==0) return Eigen::MatrixXd();
	boost::random::uniform_real_distribution<double> dist(0.0, 1.0);
	if(Dim==1U){ //Univariate case
		Eigen::VectorXd UnivariateResults(NumSamples);
		boost::math::normal NormalUnivariateSample(meanVect(0),VarCovMatrix(0,0));
		for(unsigned int j=0;j<NumSamples;j++){
			UnivariateResults(j)=boost::math::quantile(NormalUnivariateSample,dist(RandNumGen));
		}
		return UnivariateResults;
	}
	Eigen::EigenSolver<Eigen::MatrixXd> ev(VarCovMatrix);
	Eigen::EigenSolver<Eigen::MatrixXd>::EigenvalueType ComlexEigVal=ev.eigenvalues();
	Eigen::MatrixXd RealEigVal(Dim,Dim); //We know that all the eigenvalues are real so extract the real part from the EigenSolver calculation and root it and put them into a diagonal matrix
	for(unsigned int i=0;i<Dim;i++){
		for(unsigned int j=0;j<Dim;j++){
			if(i==j) RealEigVal(i,j)=sqrt(ComlexEigVal(i).real());
			else RealEigVal(i,j)=0.0;
		}
	}
	Eigen::EigenSolver<Eigen::MatrixXd>::EigenvectorsType ComlexEigVect=ev.eigenvectors();
	Eigen::MatrixXd RealEigVectors(Dim,Dim); //We know that all the eigenvectors are real so extract the real part from the EigenSolver calculation
	for(unsigned int i=0;i<Dim;i++){
		for(unsigned int j=0;j<Dim;j++){
			RealEigVectors(i,j)=ComlexEigVect(i,j).real();
		}
	}

	Eigen::MatrixXd retval=RealEigVectors*RealEigVal*RealEigVectors.transpose();

	Eigen::MatrixXd NormalExtractions(NumSamples,Dim);
	for(unsigned int i=0;i<Dim;i++){
		boost::math::normal NormalSample(meanVect(i),1);
		for(unsigned int j=0;j<NumSamples;j++){
			NormalExtractions(j,i)=boost::math::quantile(NormalSample,dist(RandNumGen));
		}
	}
	return NormalExtractions*retval;
}
double NormalDistribution::GetDensity(const Eigen::VectorXd& Coordinates) const{
	if(!AllValid || Coordinates.rows()!=Dim) return -1.0;
	if(Dim==1U){ //Univariate case
		boost::math::normal NormalUnivariate(meanVect(0),VarCovMatrix(0,0));
		return boost::math::pdf(NormalUnivariate,Coordinates(0));
	}
	double distval=(Coordinates-meanVect).transpose()*VarCovMatrix.inverse()*(Coordinates-meanVect);
	Eigen::EigenSolver<Eigen::MatrixXd>::EigenvalueType ComplexEigVal=VarCovMatrix.eigenvalues();
	double logdet=0.0;
	for(unsigned int j=0;j<Dim;j++)	logdet+=log(ComplexEigVal(j).real());
	double logretval=-((static_cast<double>(Dim)*log(2.0*boost::math::constants::pi<double>()))+logdet+distval)/2.0;
	return exp(logretval);
}
NormalDistribution::NormalDistribution(unsigned int Dimension,const Eigen::VectorXd& mVect,const Eigen::MatrixXd& CovMatr)
	:AbstarctDistribution(Dimension)
	,meanVect(mVect)
	,VarCovMatrix(CovMatr)
	,ProbToFind(0.0)
	,UseGenz(true)
	,NumSimul(500000U)
{
	CheckValidity();
}
NormalDistribution::NormalDistribution(unsigned int Dimension)
	:AbstarctDistribution(Dimension)
	,meanVect(Eigen::VectorXd(Dimension))
	,VarCovMatrix(Eigen::MatrixXd(Dimension,Dimension))
	,ProbToFind(0.0)
	,UseGenz(true)
	,NumSimul(500000U)
{
	if(Dimension>0U){
		for(unsigned int i=0;i<Dim;i++){
			meanVect(i)=0.0;
			for(unsigned int j=0;j<Dim;j++){
				if(i==j) VarCovMatrix(i,j)=1.0;
				else VarCovMatrix(i,j)=0.0;
			}
		}
	}
	CheckValidity();
}
bool NormalDistribution::CheckValidity(){
	AllValid=true;
	if(Dim<1U){ //The dimension must be at least one
		AllValid=false;
		return AllValid;
	}
	if(meanVect.rows()!=Dim){ //The mean vector must be as many elements as there are dimensions
		AllValid=false;
		return AllValid;
	}
	if(VarCovMatrix.rows()!=VarCovMatrix.cols() || VarCovMatrix.rows()!=Dim){ //The Var-Cov Matrix must be squared and have as many rows as there are dimensions
		AllValid=false;
		return AllValid;
	}
	if(VarCovMatrix!=VarCovMatrix.transpose()){ //The Var-Cov Matrix must be symmetric
		AllValid=false;
		return AllValid;
	}
	//The Variance Covariance matrix must be positive definite
	if(VarCovMatrix.determinant()<=0.0){
		AllValid=false;
		return AllValid;
	}
	Eigen::VectorXcd RelatedEigen=VarCovMatrix.eigenvalues();
	for (unsigned int i=0;i<Dim;i++){
		if(RelatedEigen(i).real()<0.0){ 
			AllValid=false;
			return AllValid;
		}
	}
	return AllValid;
}
bool NormalDistribution::SetMeanVector(const Eigen::VectorXd& mVect){
	if(mVect.rows()!=Dim) //The mean vector must be as many elements as there are dimensions
			return false;
	meanVect=mVect;
	CheckValidity();
	return true;
}
bool NormalDistribution::SetMeanVector(const std::vector<double>& mVect){
	if(mVect.size()!=Dim) return false;
	for(unsigned int i=0;i<Dim;i++){
		meanVect(i)=mVect.at(i);
	}
	CheckValidity();
	return true;
}
bool NormalDistribution::SetDimension(unsigned int Dimension){
	if(Dimension==0U) return false;
	Dim=Dimension;
	for(unsigned int i=0;i<Dim;i++){
		meanVect(i)=0.0;
		for(unsigned int j=0;j<Dim;j++){
			if(i==j) VarCovMatrix(i,j)=1.0;
			else VarCovMatrix(i,j)=0.0;
		}
	}
	CheckValidity();
	return true;
}
bool NormalDistribution::SetVarCovMatrix(const Eigen::MatrixXd& CovMatr){
	if(CovMatr.rows()!=CovMatr.cols() || CovMatr.rows()!=Dim) //The Var-Cov Matrix must be squared and have as many rows as there are dimensions
		return false;
	if(CovMatr!=CovMatr.transpose()) //The Var-Cov Matrix must be symmetric
		return false;
	//The Variance Covariance matrix must be positive definite
	if(CovMatr.determinant()<0.0)
		return false;

	Eigen::VectorXcd RelatedEigen=CovMatr.eigenvalues();
	for (unsigned int i=0;i<Dim;i++){
		if(RelatedEigen(i).real()<0.0){ 
			return false;
		}
	}
	VarCovMatrix=CovMatr;
	CheckValidity();
	return true;
}
bool NormalDistribution::SetVarCovMatrix(const std::vector<double>& mVect, bool RowWise){
	if(mVect.size()!=Dim*Dim) return false;
	Eigen::MatrixXd TempMatrix(Dim,Dim);
	for(unsigned int i=0;i<mVect.size();i++){
		TempMatrix(i/Dim,i%Dim)=mVect.at(i);
	}
	if(!RowWise) TempMatrix.transposeInPlace();
	return SetVarCovMatrix(TempMatrix);
}
bool NormalDistribution::SetVarCovMatrix(const Eigen::MatrixXd& CorrelationMatrix,const Eigen::VectorXd& Variances){
	if(CorrelationMatrix.rows()!=CorrelationMatrix.cols() || CorrelationMatrix.rows()!=Dim) return false;
	if(!CorrelationMatrix.diagonal().isOnes()) return false;
	if(Variances.rows()!=Dim) return false;
	Eigen::MatrixXd TestVariance(Dim,Dim);
	for(unsigned int i=0;i<Dim;i++){
		if(Variances(i)<0.0) return false;
		TestVariance(i,i)=Variances(i);
		for(unsigned int j=0;j<Dim;j++){
			if(i==j) continue;
			if(CorrelationMatrix(i,j)>1.0 || CorrelationMatrix(i,j)<-1.0) return false;
			else TestVariance(i,j)=CorrelationMatrix(i,j)*sqrt(abs(Variances(i)*Variances(j)));
		}
	}
	return SetVarCovMatrix(TestVariance);
}
Eigen::MatrixXd NormalDistribution::GetCorrelationMatrix() const{
	if(!AllValid) return Eigen::MatrixXd();
	Eigen::MatrixXd Result(Dim,Dim);
	for(unsigned int i=0;i<Dim;i++){
		for(unsigned int j=0;j<Dim;j++){
			if(i==j) Result(i,j)=1.0;
			else Result(i,j)=VarCovMatrix(i,j)/(sqrt(VarCovMatrix(i,i)*VarCovMatrix(j,j)));
		}
	}
	return Result;
}
std::vector<unsigned int> NormalDistribution::FillOrder(const Eigen::VectorXd& source)const{
	if(source.rows()==0 || Dim!=source.rows()) return std::vector<unsigned int>();
	std::vector<unsigned int> Result(source.rows());
	unsigned int CurrentCount;
	for(unsigned int i=0;i<Dim;i++){
		CurrentCount=0;
		for(unsigned int j=0;j<Dim;j++){
			if(source(j)<source(i)) CurrentCount++;
		}
		Result[i]=CurrentCount;
	}
	for(unsigned int i=0;i<Dim;i++){
		for(unsigned int j=0;j<i;j++){
			if(Result[i]==Result[j]) Result[i]++;
		}
	}
	return Result;
}
NormalDistribution::NormalDistribution(const NormalDistribution& a)
	:meanVect(a.meanVect)
	,VarCovMatrix(a.VarCovMatrix)
	,ProbToFind(a.ProbToFind)
	,UseGenz(a.UseGenz)
	,NumSimul(a.NumSimul)
{
	AllValid=a.AllValid;
	Dim=a.Dim;
	CurrentSeed=a.CurrentSeed;
	RandNumGen.seed(CurrentSeed);
}
NormalDistribution& NormalDistribution::operator=(const NormalDistribution& a){
	AllValid=a.AllValid;
	Dim=a.Dim;
	meanVect=a.meanVect;
	VarCovMatrix=a.VarCovMatrix;
	ProbToFind=a.ProbToFind;
	UseGenz=a.UseGenz;
	NumSimul=a.NumSimul;
	return *this;
}
boost::math::tuple<double, double> NormalDistribution::operator()(double x){
	Eigen::VectorXd CoordinatesVector(Dim);
	for(unsigned i=0;i<Dim;i++) CoordinatesVector(i)=x;
	return boost::math::make_tuple(GetCumulativeDesity(CoordinatesVector)-ProbToFind,GetDensity(CoordinatesVector));
}
Eigen::MatrixXd NormalDistribution::ExtractSamplesCDF(unsigned int NumSamples) const{
	if(!AllValid || NumSamples<1U) return Eigen::MatrixXd();
	Eigen::MatrixXd Result=ExtractSamples(NumSamples);
	for(unsigned int j=0;j<Dim;j++){
		boost::math::normal NormalDist(meanVect(j),VarCovMatrix(j,j));
		for(unsigned int i=0;i<Dim;i++){
			Result(i,j)=boost::math::cdf(NormalDist,Result(i,j));
		}
	}
	return Result;
}
bool NormalDistribution::SetNumSimul(unsigned int a){
	if(a>0) NumSimul=a;
	return a>0;
}

#ifdef mvNormSamplerUnsafeMethods
void NormalDistribution::SetMeanVector(double* mVect){
	for(unsigned int i=0;i<Dim;i++){
		meanVect(i)=mVect[i];
	}
}
bool NormalDistribution::SetVarCovMatrix(double** mVect){
	Eigen::MatrixXd TempMatrix(Dim,Dim);
	for(unsigned int i=0;i<Dim;i++){
		for(unsigned int j=0;j<Dim;j++){
			TempMatrix(i,j)=mVect[i][j];
		}
	}
	return SetVarCovMatrix(TempMatrix);
}
#endif