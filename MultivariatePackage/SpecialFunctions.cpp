#include "SpecialFunctions.h"
#include <boost/math/distributions/gamma.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/exponential_distribution.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/random/mersenne_twister.hpp>
double Multivariate::GammaFunction(double t){
	if(t<=0) return 0.0;
	boost::math::gamma_distribution<double> GammaDist(t,1.0);
	return exp(-1.0)/pdf(GammaDist,1.0);
}
double Multivariate::SimulateStable(double Alpha,double Beta,double Gamma,double Delta,unsigned int RandNumGenSeed){
	if(Alpha<=0.0 || Alpha>2.0 || Beta<-1.0 || Beta>1.0 || Gamma<=0.0) return 0.0;
	boost::random::mt19937 RandNumGen;
	if(RandNumGenSeed==0) RandNumGen.seed(static_cast<unsigned int>(std::time(NULL)));
	else RandNumGen.seed(RandNumGenSeed);
	boost::random::uniform_real_distribution<double> dist(0.0,1.0);
	boost::random::uniform_real_distribution<double> Unif(-boost::math::constants::pi<double>()/2.0, boost::math::constants::pi<double>()/2.0);
	boost::random::exponential_distribution<double> ExpDistr(1.0);
	const double Sim1=Unif(RandNumGen);
	const double Sim2=ExpDistr(RandNumGen);
	const double temp1=atan(Beta*tan(Alpha*boost::math::constants::pi<double>()/2.0))/Alpha;
	double temp2;
	if(Alpha==1.0)
		temp2=(2.0/boost::math::constants::pi<double>())*(
		(
		((2.0/boost::math::constants::pi<double>())+(Beta*Sim1))*tan(Sim1)
		) - (
		Beta*log(
		(cos(Sim1)*Sim2*boost::math::constants::pi<double>()/2.0)
		/
		((boost::math::constants::pi<double>()/2.0)+(Beta*Sim1))
		)
		)
		);
	else
		temp2=((sin(Alpha)*(temp1+Sim1)) /
		pow(cos(Alpha)*temp1*cos(Sim1),1.0/Alpha)) *
		pow(cos((Alpha*temp1)+((Alpha-1.0)*Sim1))/Sim2,(1.0-Alpha)/Alpha);
	if(Alpha==1.0)
		return (Gamma*temp2)+Delta+(Beta*Gamma*log(Gamma)*boost::math::constants::pi<double>()/2.0);
	else
		return (Gamma*temp2)+Delta;
}