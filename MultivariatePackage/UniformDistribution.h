#ifndef UniformDistribution_h__
#define UniformDistribution_h__
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <boost/random/mersenne_twister.hpp>
namespace Multivariate{
	class UniformDistribution{
	private:
		mutable boost::random::mt19937 RandNumGen;
		Eigen::Matrix<double,-1,2> Limits;
		unsigned int Dim;
		bool AllValid;
		unsigned int CurrentSeed;
	public:
		UniformDistribution(unsigned int Dimension=1U);
		UniformDistribution(unsigned int Dimension, const Eigen::Matrix<double,-1,2>& MinMax);
		bool IsValid() const {return AllValid;}
		void SetRandomSeed(unsigned int NewSeed);
		unsigned int GetCurrentSeed()const{return CurrentSeed;}
		bool SetDimension(unsigned int Dimension);
		unsigned int GetDimension() const {return Dim;}
		const Eigen::Matrix<double,-1,2>& GetLimits() const {return Limits;}
		bool SetLimits(const Eigen::Matrix<double,-1,2>& MinMax);
		Eigen::RowVectorXd ExtractSample() const{return ExtractSamples(1U);}
		std::vector<double> ExtractSampleVector() const;
		Eigen::MatrixXd ExtractSamples(unsigned int NumSamples) const;
		std::map<unsigned int,std::vector<double> > ExtractSamplesMap(unsigned int NumSamples) const;
		double GetDensity(bool GetLogDensity=false)const;
		double GetDensity(bool GetLogDensity=false)const;
		double GetCumulativeDesity(const std::vector<double>& Coordinates)const;
		double GetCumulativeDesity(const Eigen::VectorXd& Coordinates)const;
		Eigen::VectorXd GetQuantile(double Prob);
		std::vector<double> GetQuantileVector(double Prob);
	};
	class IndependenceCopula : public UniformDistribution{
		private:
			IndependenceCopula(unsigned int Dimension, const Eigen::Matrix<double,-1,2>& MinMax) : UniformDistribution(Dimension,MinMax){}
			bool SetLimits(const Eigen::Matrix<double,-1,2>& MinMax){return UniformDistribution::SetLimits(MinMax);}
		public:
			IndependenceCopula(unsigned int Dimension=1U) : UniformDistribution(Dimension){}
	};
}
#endif // UniformDistribution_h__
