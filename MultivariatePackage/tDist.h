#include <Eigen/Dense>
#include <vector>
#include <map>
#include <boost/random/mersenne_twister.hpp>
namespace Multivariate{
	class tDistribution{
	private:
		mutable boost::random::mt19937 RandNumGen;
		unsigned int Dim;
		Eigen::VectorXd LocationVect;
		Eigen::MatrixXd ScaleMatrix;
		unsigned int DegreesOfFreedom;
		bool AllValid;
		bool CheckValidity();
		tDistribution(const tDistribution& a);
		tDistribution& operator=(const tDistribution& a);
		unsigned int CurrentSeed;
	public:
		tDistribution(unsigned int Dimension,unsigned int DegFreedom,const Eigen::VectorXd& locVect,const Eigen::MatrixXd& ScaleMatr);
		tDistribution(unsigned int Dimension=1U,unsigned int DegFreedom=1U);
		bool IsValid() const {return AllValid;}
		void SetRandomSeed(unsigned int NewSeed);
		unsigned int GetCurrentSeed()const{return CurrentSeed;}
		bool SetLocationVector(const Eigen::VectorXd& mVect);
		bool SetLocationVector(const std::vector<double>& mVect);
		bool SetDimension(unsigned int Dimension);
		bool SetScaleMatrix(const Eigen::MatrixXd& CovMatr);
		bool SetScaleMatrix(const std::vector<double>& mVect, bool RowWise=true);
		bool SetDegreesOfFreedom(unsigned int a);
		unsigned int GetDimension() const {return Dim;}
		const Eigen::VectorXd& GetLocationVector() const {return LocationVect;}
		const Eigen::MatrixXd& GetScaleMatrix() const {return ScaleMatrix;}
		unsigned int GetDegreesOfFreedom() const {return DegreesOfFreedom;}
		Eigen::RowVectorXd ExtractSample() const{return ExtractSamples(1U);}
		std::vector<double> ExtractSampleVector() const;
		Eigen::MatrixXd ExtractSamples(unsigned int NumSamples) const;
		std::map<unsigned int,std::vector<double> > ExtractSamplesMap(unsigned int NumSamples) const;
		double GetDensity(const Eigen::VectorXd& Coordinates, bool GetLogDensity=false)const;
		double GetDensity(const std::vector<double>& Coordinates, bool GetLogDensity=false)const;
		double GetCumulativeDesity(const std::vector<double>& Coordinates, bool UseGenz=true, unsigned int NumSimul=500000)const;
		double GetCumulativeDesity(const Eigen::VectorXd& Coordinates, bool UseGenz=true, unsigned int NumSimul=500000)const;
	};
}