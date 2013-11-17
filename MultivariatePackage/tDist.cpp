#include "tDist.h"
#include "NormalDist.h"
#include <ctime>
using namespace Multivariate;
tDistribution::tDistribution(unsigned int Dimension,unsigned int DegFreedom,const Eigen::VectorXd& locVect,const Eigen::MatrixXd& ScaleMatr)
	:Dim(Dimension)
	,DegreesOfFreedom(DegFreedom)
	,LocationVect(locVect)
	,ScaleMatrix(ScaleMatr)
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