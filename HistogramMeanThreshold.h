//////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2013 Carlos Becker                                             //
// Ecole Polytechnique Federale de Lausanne                                     //
// Contact <carlos.becker@epfl.ch> for comments & bug reports                   //
//                                                                              //
// This program is free software: you can redistribute it and/or modify         //
// it under the terms of the version 3 of the GNU General Public License        //
// as published by the Free Software Foundation.                                //
//                                                                              //
// This program is distributed in the hope that it will be useful, but          //
// WITHOUT ANY WARRANTY; without even the implied warranty of                   //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU             //
// General Public License for more details.                                     //
//                                                                              //
// You should have received a copy of the GNU General Public License            //
// along with this program. If not, see <http://www.gnu.org/licenses/>.         //
//////////////////////////////////////////////////////////////////////////////////

#ifndef HISTOGRAMMEANTHRESHOLD_H
#define HISTOGRAMMEANTHRESHOLD_H

#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <cstdio>

#include "IntegralImage.h"
#include <libconfig.hh>

// a single box for each location
typedef BoxPosition                     BoxPositionType;


// for a list of samples, we need a list of BoxPositionType elements, one for each
typedef std::vector<BoxPositionType>    BoxPositionVector;

typedef double AdaBoostErrorType;

#define USE_DIFFERENCE_SPLIT	0

struct HistogramMeanThresholdData
{
	// number of weak learners to explore (train)
	unsigned int numWLToExplore;
	
	
    const std::vector<UIntPoint3D>    &svCentroid; // SV centroids
    const std::vector< Eigen::Matrix3f >  &rotMatrices;

    const std::vector<IntegralImage<IntegralImageType> * > &pixIntImages; // list of integral images of features

    float  distMin, distMax;
    unsigned int distSteps;

    float  angThMin, angThMax;
    unsigned int angThSteps;
    
    float  angPhMin, angPhMax;
    unsigned int angPhSteps;
    
    float zAnisotropyFactor;
    float invZAnisotropyFactor; // a cached version of 1.0 / zAnisotropyFactor

    unsigned int rMin;
    unsigned int rMax;
    unsigned int rSteps;


    Eigen::Matrix< float, 3, Eigen::Dynamic >  possibleOffsets;

    std::vector<unsigned int>       possibleOffsetsRadius;

    const Matrix3D<unsigned int>      &pixToSV; //pixel to supervoxel map

#if APPLY_PATCH_VARNORM
    // inverse std dev (1/stddev) and mean for every channel, per supervoxel
    const std::vector< std::vector<float> >  &svoxWindowInvStd;
    const std::vector< std::vector<float> >  &svoxWindowMean;
#endif

    inline HistogramMeanThresholdData( const std::vector<UIntPoint3D> &centroids,
                                const std::vector< Eigen::Matrix3f > &rMatrices,
                                const std::vector<IntegralImage<IntegralImageType> * >  &iimgs,
                                const Matrix3D<unsigned int> &pixtosv,
                                const double _zAnisotropyFactor
                        #if APPLY_PATCH_VARNORM
                                ,
                                const std::vector< std::vector<float> > &_svoxWindowInvStd,
                                const std::vector< std::vector<float> > &_svoxWindowMean
                        #endif
                                       )
        : svCentroid(centroids), rotMatrices(rMatrices), pixIntImages(iimgs), pixToSV(pixtosv),
        zAnisotropyFactor(_zAnisotropyFactor)
        #if APPLY_PATCH_VARNORM
        ,svoxWindowInvStd(_svoxWindowInvStd), svoxWindowMean(_svoxWindowMean)
        #endif
    {
        invZAnisotropyFactor = 1.0 / zAnisotropyFactor;
        
		numWLToExplore = 4000;
		
#if !LOCAL_ONLY
        #pragma message("Using remote cubes")
        distMin = -20;
        distMax = 20;
        distSteps = 11;

        rMin = 0;
        rMax = 20;
        rSteps = 11;

        angThMin = 0;
        angThMax = M_PI/2 - 0.1;
        angThSteps =  6;
        //angThSteps =  3;

        angPhMin = 0.0;
        angPhMax = 2*M_PI;
        angPhSteps = 10;
        //angPhSteps =  5;
#else
        #pragma message("Using local cubes only")
        distMin = 0;
        distMax = 0;
        distSteps = 1;

        rMin = 0;
        rMax = 20;
        rSteps = 11;

        angThMin = 0;
        angThMax = 0;
        angThSteps =  1;
        //angThSteps =  3;

        angPhMin = 0.0;
        angPhMax = 0.0;
        angPhSteps = 1;

#endif

#if 0
        rMin = 0;
        rMax = 20;
        rSteps = 11;

        distMin = distMax = 0;
        distSteps = 1;

        angThMin = 0;
        angThMax = 0;
        angThSteps = 1;

        angPhMin = 0;
        angPhMax = 0;
        angPhSteps = 1;
#endif

        precomputeOrients();
    }

    void precomputeOrients()
    {
        possibleOffsetsRadius.clear();

		unsigned rStep = 2;

        possibleOffsets.resize( 3, distSteps * angThSteps * angPhSteps * rSteps );
        possibleOffsetsRadius.resize( possibleOffsets.size() );

        double dStep = (distMax - distMin) / std::max( (distSteps - 1), 1U );
        double angThStep = (angThMax - angThMin) / std::max( (angThSteps - 1), 1U );
        double angPhStep = (angPhMax - angPhMin) / std::max( (angPhSteps - 1), 1U );

        qDebug("dStep: %f", dStep);
        qDebug("angThStep: %f", angThStep);
        qDebug("angPhStep: %f", angPhStep);
        
        unsigned ii = 0;
        for (unsigned ths=0; ths < angThSteps; ths++)
        {
			double theta = angThMin + angThStep * ths;
			
			for (unsigned phs=0; phs < angPhSteps; phs++)
			{
				double phi = angPhMin + angPhStep * phs;
				
				Eigen::Vector3f angleVec;
				angleVec(0) = cos(phi)*sin(theta);
				angleVec(1) = sin(phi)*sin(theta);
				angleVec(2) = cos(theta);
				std::cout << "Angvec: " << angleVec << std::endl;
				
				for (unsigned ds=0; ds < distSteps; ds++)
				{
					float d = distMin + dStep*ds;

                    for (unsigned r = rMin; r <= rMax; r += rStep)
                    {
                        possibleOffsets.col(ii) = d * angleVec;
                        possibleOffsetsRadius[ii] = r;

                        ii++;
                    }
				}
			}
		}

		qDebug("---> Number of pose indexes: %d (%d)", (int) possibleOffsets.cols(), (int)ii );
            if (possibleOffsets.cols() != ii)
                qFatal("error here, fix: %d %d!", (int)possibleOffsets.cols(), (int)ii );
    }

    float poseIndexMagnitude( unsigned poseIdx ) const
    {
        if (poseIdx >= possibleOffsets.cols())
            qFatal("Poseidx exceed possible value");
        return possibleOffsets.col(poseIdx).norm();
    }
    
    float poseIndexTheta( unsigned poseIdx ) const
    {
        if (poseIdx >= possibleOffsets.cols())
            qFatal("Poseidx exceed possible value");
        return 180.0 * acos( possibleOffsets.col(poseIdx)(2) / possibleOffsets.col(poseIdx).norm() ) / M_PI;
	}
	
    float poseIndexPhi( unsigned poseIdx ) const
    {
        if (poseIdx >= possibleOffsets.cols())
            qFatal("Poseidx exceed possible value");
            
        return 180.0 * atan2( possibleOffsets.col(poseIdx)(1), possibleOffsets.col(poseIdx)(0) ) / M_PI;
    }

    unsigned poseIndexRadius( unsigned poseIdx ) const
    {
        if (poseIdx >= possibleOffsetsRadius.size())
            qFatal("Poseidx exceed possible value");

        return possibleOffsetsRadius[poseIdx];
    }

    inline void setDistanceLimits( float min, float max ) {
        distMin = min;
        distMax = max;
    }

    // returns number of pose idxs
    inline unsigned numPosIdx() const
    {
        return possibleOffsets.cols();
    }

    void poseIndexedFeature( const unsigned poseIdx, const unsigned sampleIdx, BoxPositionType *box ) const
    {
        /*if (poseIdx >= possibleOffsetsRadius.size())
            qFatal("Poseidx exceed possible value");*/

        const unsigned int Vwidth = pixToSV.width();
        const unsigned int Vheight = pixToSV.height();
        const unsigned int Vdepth = pixToSV.depth();

        const float radius = possibleOffsetsRadius[poseIdx];

        //for (unsigned int s=0; s < sampleIdxs.size(); s++)
        {
            UIntPoint3D  pt = svCentroid[ sampleIdx ];

            // resize first
            Eigen::Vector3f orient = rotMatrices[sampleIdx] * possibleOffsets.col(poseIdx);
            FloatPoint3D newPt;
            
            newPt.x = round(pt.x + orient.coeff(0));
            newPt.y = round(pt.y + orient.coeff(1));
            newPt.z = round(pt.z + invZAnisotropyFactor * orient.coeff(2));

            //FIXME //TODO quick bug patch
           if (isnan(newPt.x)) newPt.x = 1;
           if (isnan(newPt.y)) newPt.y = 1;
           if (isnan(newPt.z)) newPt.z = 1;

            if (newPt.x < 1)    newPt.x = 1;
            if (newPt.y < 1)    newPt.y = 1;
            if (newPt.z < 1)    newPt.z = 1;

            if (newPt.x >= Vwidth)    newPt.x = Vwidth - 1;
            if (newPt.y >= Vheight)   newPt.y = Vheight - 1;
            if (newPt.z >= Vdepth)    newPt.z = Vdepth - 1;

            float rx = radius;
            float ry = radius;
            float rz = invZAnisotropyFactor * radius;

            // check image borders
            if ( newPt.x - rx <= 1 ) rx = newPt.x - 2;
            if ( newPt.y - ry <= 1 ) ry = newPt.y - 2;
            if ( newPt.z - rz <= 1 ) rz = newPt.z - 2;

            if ( newPt.x + rx >= Vwidth )  rx =  newPt.x - Vwidth - 1;
            if ( newPt.y + ry >= Vheight)  ry =  newPt.y - Vheight - 1;
            if ( newPt.z + rz >= Vdepth )  rz =  newPt.z - Vdepth - 1;

            rx = (rx >= 0)? rx : 0;
            ry = (ry >= 0)? ry : 0;
            rz = (rz >= 0)? rz : 0;

            box->x = newPt.x;
            box->y = newPt.y;
            box->z = newPt.z;

            box->rx = rx;
            box->ry = ry;
            box->rz = rz;
        }
    }
};

#if 0   // needs to be fixed
template<typename ParamsType, typename WeightsType, bool cacheValues = true>
struct FeatureSubtractorOperator
{
    const unsigned mPoseIdxA, mPoseIdxB;
	const WeightsType	  &mWeights;
	const std::vector<unsigned int> &mClassLabels;
	const unsigned int mCol;
        const ParamsType &mParams;

        std::vector< IntegralImageType > mCachedValues;
	
        inline FeatureSubtractorOperator( const unsigned poseIdxA, const unsigned poseIdxB,
			const WeightsType &weights, const std::vector<unsigned int> classLabels,
                        unsigned int numCol, const ParamsType &params ):
                        mPoseIdxA(poseIdxA), mPoseIdxB(poseIdxB), mWeights(weights), mClassLabels(classLabels),
                        mCol(numCol), mParams(params)
        {
            if (cacheValues)
            {
                mCachedValues.resize( mWeights.size() );

                for (unsigned i=0; i < mWeights.size(); i++)
                    mCachedValues[i] = nonCachedValue(i);
            }
        }
	
        inline IntegralImageType nonCachedValue( unsigned int idx ) const 
        {
            BoxPositionType boxA, boxB;
            mParams.poseIndexedFeature( mPoseIdxA, mSampleIdxs
            
            return mParams.pixIntImages[mCol]->centeredSum( boxA ) - mParams.pixIntImages[mCol]->centeredSum( boxB );
        }

        inline IntegralImageType value( unsigned int idx ) const {
            if (cacheValues)
                return mCachedValues[idx];
            else
                return nonCachedValue(idx);
	}
	
	inline typename WeightsType::Scalar weight( unsigned int idx ) const {
		return mWeights.coeff( idx );
	}
	
	inline unsigned int label( unsigned int idx ) const {
		return mClassLabels[idx];
	}

        inline unsigned int count() const { return mWeights.size(); }
};
#endif

template<typename ParamsType, typename WeightsType, bool cacheValues = true>
struct FeatureRawOperator
{
    const unsigned mPoseIdx;
	const WeightsType	  &mWeights;
	const std::vector<unsigned int> &mClassLabels;
	const unsigned int mCol;
    const ParamsType &mParams;
    const std::vector<unsigned int> &sampleIdxs;
    std::vector< IntegralImageType > mCachedValues;
	
        inline FeatureRawOperator( const unsigned poseIdx, const WeightsType &weights, const std::vector<unsigned int> classLabels,
                        unsigned int numCol, const ParamsType &params,
                        const std::vector<unsigned int> &_sampleIdxs ):
                        mPoseIdx(poseIdx), mWeights(weights), mClassLabels(classLabels),
                        mCol(numCol), mParams(params),
                        sampleIdxs(_sampleIdxs)
        {
            if (cacheValues)
            {
                mCachedValues.resize( mWeights.size() );

                for (unsigned i=0; i < mWeights.size(); i++)
                    mCachedValues[i] = nonCachedValue(i);
            }
        }
	
        inline IntegralImageType nonCachedValue( unsigned int idx ) const
        {
            BoxPositionType box;
            
            mParams.poseIndexedFeature( mPoseIdx, sampleIdxs[idx], &box );
            
            #if APPLY_PATCH_VARNORM
                return mParams.pixIntImages[mCol]->centeredSumNormalized( box, mParams.svoxWindowMean[mCol][sampleIdxs[idx]], mParams.svoxWindowInvStd[mCol][sampleIdxs[idx]]);
            #else
                return mParams.pixIntImages[mCol]->centeredSum( box );
            #endif
        }

        inline IntegralImageType value( unsigned int idx ) const
        {
            if (cacheValues)
                return mCachedValues[idx];
            else
                return nonCachedValue(idx);
        }
	
	inline typename WeightsType::Scalar weight( unsigned int idx ) const {
		return mWeights.coeff( idx );
	}
	
	inline unsigned int label( unsigned int idx ) const {
		return mClassLabels[idx];
	}
	
    inline unsigned int count() const { return mWeights.size(); }
};

template<typename ValuesArrayType, typename WeightsType>
struct FeatureOperatorPrecomputedValues
{
    const WeightsType     &mWeights;
    const std::vector<unsigned int> &mClassLabels;
    const std::vector<unsigned int> &sampleIdxs;
    const ValuesArrayType &mCachedValues;
    
    typedef typename ValuesArrayType::Scalar ValueScalarType;
    
    inline FeatureOperatorPrecomputedValues( 
                    const ValuesArrayType &cachedValues,
                    const WeightsType &weights, const std::vector<unsigned int> classLabels,
                    const std::vector<unsigned int> &_sampleIdxs ):
                    mCachedValues(cachedValues), mWeights(weights), mClassLabels(classLabels),
                    sampleIdxs(_sampleIdxs)
    {
    }

    inline ValueScalarType value( unsigned int idx ) const
    {
        return mCachedValues.coeff(idx);
    }
    
    inline typename WeightsType::Scalar weight( unsigned int idx ) const {
        return mWeights.coeff( idx );
    }
    
    inline unsigned int label( unsigned int idx ) const {
        return mClassLabels[idx];
    }
    
    inline unsigned int count() const { return mWeights.size(); }
};

template<typename FeatureOperatorType>
class SortFeature
{
private:
    const FeatureOperatorType &mFop;

public:
    inline SortFeature( const FeatureOperatorType &fop ) : mFop(fop) {}

    inline bool operator()(unsigned int l, unsigned int r) const
    {
        return mFop.value(l) < mFop.value(r);
    }
};

template<typename T2>
class SortMatrixByColumnOnlyIndex
{
private:
    const T2 &mData;

    const unsigned int mCol;

public:
    SortMatrixByColumnOnlyIndex( const T2 &data, unsigned int col ) : mData(data), mCol(col) {}

    inline bool operator()(unsigned int l, unsigned int r) const
    {
        return mData(l, mCol) < mData(r,mCol);
    }
};


template<typename T>
static std::vector<T> genIndexSequence( T N )
{
    std::vector<T>  vec(N);
    for (unsigned int i=0; i < N; i++)
        vec[i] = i;

    return vec;
}

class ThresholdSplit
{
private:
    IntegralImageType mThreshold;
    bool  mInvert;
    unsigned int mColumn;
    unsigned int mPoseIdx;
    std::string mDescription;

public:

    ThresholdSplit() {}
    ThresholdSplit( unsigned int column, IntegralImageType threshold, bool invert, unsigned int poseIdx ) {
        mThreshold = threshold;
        mInvert = invert;
        mColumn = column;
        mPoseIdx = poseIdx;
    }

    unsigned int poseIdx() const { return mPoseIdx; }

    void save( libconfig::Setting &s ) const
    {
        s.add("threshold", libconfig::Setting::TypeFloat) = mThreshold;
        s.add("invert", libconfig::Setting::TypeBoolean) = mInvert;
        s.add("column", libconfig::Setting::TypeInt) = (int)mColumn;
        s.add("poseIdx", libconfig::Setting::TypeInt) = (int)mPoseIdx;
        s.add("description", libconfig::Setting::TypeString) = mDescription;
    }

    void load( const libconfig::Setting &s )
    {
        mThreshold = s["threshold"];
        mInvert = s["invert"];
        mColumn = (int) s["column"];
        mPoseIdx = (int) s["poseIdx"];
        mDescription = (const char *) s["description"];
        /*s.add("invert", libconfig::Setting::TypeBool) = mInvert;
        s.add("column", libconfig::Setting::TypeInt) = mColumn;
        s.add("poseIdx", libconfig::Setting::TypeInt) = mPoseIdx;
        s.add("description", libconfig::Setting::TypeString) = mDescription;*/
    }

    IntegralImageType threshold() const { return mThreshold; }
    bool invert() const { return mInvert; }
    unsigned int column() const { return mColumn; }
    void invertClassifier() { mInvert = !mInvert; }
    
    void setInvert( bool yes ) { mInvert = yes; }
    void setThreshold( IntegralImageType thr ) { mThreshold = thr; }

    inline void setStringDescrption( const std::string &descr )
    {
        mDescription = descr;
    }

    inline const std::string &getStringDescription() const
    {
        return mDescription;
    }
    
    template<typename SampleIdxVector, typename MatrixType>
    void exportFeat( const SampleIdxVector &sampleIdxs, const HistogramMeanThresholdData &params, MatrixType &destMat, unsigned int col )
    {
		#pragma omp parallel for
        for (unsigned int i=0; i < sampleIdxs.size(); i++)
        {
            BoxPositionType   box;
            params.poseIndexedFeature( mPoseIdx, sampleIdxs[i], &box );
            
            #if APPLY_PATCH_VARNORM
                destMat.coeffRef( i, col ) = params.pixIntImages[mColumn]->centeredSumNormalized( box, params.svoxWindowMean[mColumn][sampleIdxs[i]], params.svoxWindowInvStd[mColumn][sampleIdxs[i]] );
            #else
                destMat.coeffRef( i, col ) = params.pixIntImages[mColumn]->centeredSum( box );
            #endif
        }
    }


    template<typename SampleIdxVector, typename PredType>
    void classify( const SampleIdxVector &sampleIdxs, const HistogramMeanThresholdData &params, std::vector<PredType> &prediction, const unsigned numThreads = 1 )
    {
        prediction.resize( sampleIdxs.size() );
        classifyLowLevel<SampleIdxVector, PredType, 0>( sampleIdxs, params, prediction.data(), numThreads );
    }

    template<typename SampleIdxVector, int negativeValue>
    void classify( const SampleIdxVector &sampleIdxs, const HistogramMeanThresholdData &params, Eigen::ArrayXf &prediction, const unsigned numThreads = 1 )
    {
        prediction.resize( sampleIdxs.size() );
        classifyLowLevel<SampleIdxVector, float, negativeValue>( sampleIdxs, params, prediction.data(), numThreads );
    }

    template<typename SampleIdxVector, typename PredType, int negativeValue>
    void classifyLowLevel( const SampleIdxVector &sampleIdxs, const HistogramMeanThresholdData &params, PredType *prediction, const unsigned numThreads = 1 )
    {
        if (mInvert == false)
        {
            #pragma omp parallel for num_threads(numThreads)
            for (unsigned int i=0; i < sampleIdxs.size(); i++)
            {
                BoxPositionType box;
                params.poseIndexedFeature( mPoseIdx, sampleIdxs[i], &box );
                
                #if APPLY_PATCH_VARNORM
                if ( params.pixIntImages[mColumn]->centeredSumNormalized( box, params.svoxWindowMean[mColumn][sampleIdxs[i]], params.svoxWindowInvStd[mColumn][sampleIdxs[i]]  ) >= mThreshold )
                #else
                if ( params.pixIntImages[mColumn]->centeredSum( box ) >= mThreshold )
                #endif
                    prediction[i] = 1;
                else
                    prediction[i] = negativeValue;
            }
        } else {
            #pragma omp parallel for num_threads(numThreads)
            for (unsigned int i=0; i < sampleIdxs.size(); i++)
            {
                BoxPositionType box;
                params.poseIndexedFeature( mPoseIdx, sampleIdxs[i], &box );
                
                #if APPLY_PATCH_VARNORM
                if ( params.pixIntImages[mColumn]->centeredSumNormalized( box, params.svoxWindowMean[mColumn][sampleIdxs[i]], params.svoxWindowInvStd[mColumn][sampleIdxs[i]]  ) < mThreshold ) {
                #else
                if ( params.pixIntImages[mColumn]->centeredSum( box ) < mThreshold ) {
                #endif
                    prediction[i] = 1;
                } else {
                    prediction[i] = negativeValue;
                }
            }
        }
    }
};


class ThresholdSplitSubtract
{
private:
    IntegralImageType mThreshold;
    bool  mInvert;
    unsigned int mColumn;
    unsigned int mPoseIdxA, mPoseIdxB;
    std::string mDescription;

public:
    ThresholdSplitSubtract() {}
    ThresholdSplitSubtract( unsigned int column, IntegralImageType threshold, bool invert, unsigned int poseIdxA, unsigned int poseIdxB ) {
        mThreshold = threshold;
        mInvert = invert;
        mColumn = column;
        
        mPoseIdxA = poseIdxA;
        mPoseIdxB = poseIdxB;
    }

    inline void setStringDescrption( const std::string &descr )
    {
        mDescription = descr;
    }

    inline const std::string &getStringDescription() const
    {
        return mDescription;
    }

    IntegralImageType threshold() const { return mThreshold; }
    bool invert() const { return mInvert; }
    unsigned int column() const { return mColumn; }
    
    template<typename SampleIdxVector, typename MatrixType>
    void exportFeat( const SampleIdxVector &sampleIdxs, const HistogramMeanThresholdData &params, MatrixType &destMat, unsigned int col )
    {
        for (unsigned int i=0; i < sampleIdxs.size(); i++)
        {
            BoxPositionType   boxA, boxB;
            params.poseIndexedFeature( mPoseIdxA, sampleIdxs[i], &boxA );
            params.poseIndexedFeature( mPoseIdxB, sampleIdxs[i], &boxB );
            
            destMat.coeffRef( i, col ) = params.pixIntImages[mColumn]->centeredSum( boxA ) - params.pixIntImages[mColumn]->centeredSum( boxB );
        }
    }

    template<typename SampleIdxVector, typename PredType>
    void classify( const SampleIdxVector &sampleIdxs, const HistogramMeanThresholdData &params, std::vector<PredType> &prediction )
    {
        prediction.resize( sampleIdxs.size() );

        if (mInvert == false)
        {
            for (unsigned int i=0; i < sampleIdxs.size(); i++)
            {
                BoxPositionType   boxA, boxB;
                params.poseIndexedFeature( mPoseIdxA, sampleIdxs[i], &boxA );
                params.poseIndexedFeature( mPoseIdxB, sampleIdxs[i], &boxB );
                
                if ( params.pixIntImages[mColumn]->centeredSum( boxA ) - params.pixIntImages[mColumn]->centeredSum( boxB ) >= mThreshold )
                    prediction[i] = 1;
                else
                    prediction[i] = 0;
            }
        } else {
            for (unsigned int i=0; i < sampleIdxs.size(); i++)
            {
                BoxPositionType   boxA, boxB;
                params.poseIndexedFeature( mPoseIdxA, sampleIdxs[i], &boxA );
                params.poseIndexedFeature( mPoseIdxB, sampleIdxs[i], &boxB );
                
                if ( params.pixIntImages[mColumn]->centeredSum( boxA ) - params.pixIntImages[mColumn]->centeredSum( boxB ) < mThreshold ) {
                    prediction[i] = 1;
                } else {
                    prediction[i] = 0;
                }
            }
        }
    }
};

template<typename FeatureOperatorType>
AdaBoostErrorType computeError(  const FeatureOperatorType &fOp,
                     IntegralImageType threshold, bool inverted )
{
    AdaBoostErrorType err = 0;

    if (inverted == false)
    {
        for (unsigned int i=0; i < fOp.count(); i++)
        {
            if ( fOp.value(i) >= threshold ) {
                if ( fOp.label(i) == 0 )
                    err += fOp.weight(i);
            } else {
                if ( fOp.label(i) != 0 )
                    err += fOp.weight(i);
            }
        }
    } else {
        for (unsigned int i=0; i < fOp.count(); i++)
        {
            if ( fOp.value(i) < threshold ) {
                if ( fOp.label(i) == 0 )
                    err += fOp.weight(i);
            } else {
                if ( fOp.label(i) != 0 )
                    err += fOp.weight(i);
            }
        }
    }

    return err;
}

// if forcePolarity = true, then it will use the polarity found in 'inv'
template<typename FeatureOperatorType, bool TForcePolarity>
inline AdaBoostErrorType findBestThreshold( const FeatureOperatorType &fOp,
                        IntegralImageType &retThr, bool &inv )
{
	const unsigned int N = fOp.count();
	
    std::vector<unsigned int> sortedIdx(N);  //alloc
    for (unsigned int i=0; i < N; i++)
        sortedIdx[i] = i;

    std::sort( sortedIdx.begin(), sortedIdx.end(),
               SortFeature<FeatureOperatorType>( fOp ) );

    // find out the error if we put threshold on zero
    AdaBoostErrorType minErr = 1e6;
    IntegralImageType bestThr = 0;
    bool bestInv = false;
    unsigned int bestIdx = 0;


    // pre-compute errors with threshold on first element
    IntegralImageType firstThr = fOp.value( sortedIdx[0] );
    AdaBoostErrorType errInv = computeError( fOp, firstThr, true );
    AdaBoostErrorType err = computeError( fOp, firstThr, false );

    bestInv = errInv < err;
    bestThr = firstThr;
    
    if (TForcePolarity)
    {
        bestInv = inv;
    }
    
    if (bestInv)
        minErr = errInv;
    else
        minErr = err;

#if 0
    if (fOp.mBoxes[0].r == 0) {
        qDebug("Val %d: %f", (int)fOp.mCol, fOp.value(12));
    }
#endif
        
    //qDebug("St: %f %f %f", err, errInv, firstThr);

    std::vector<IntegralImageType>  uniqueThresholds;
    uniqueThresholds.push_back( firstThr );

    for (unsigned int i=1; i < N; i++)
    {
        const unsigned int sIdxThis = sortedIdx[i];
        const unsigned int sIdxPrev = sortedIdx[i-1];

        IntegralImageType thr = fOp.value( sIdxThis );
        IntegralImageType prevThr = fOp.value( sIdxPrev );

        // not inv => changes if
#if 1
        if ( fOp.label(sIdxPrev) == 1 ) {
            err += fOp.weight(sIdxPrev);
            errInv -= fOp.weight(sIdxPrev);
        }
        else if ( fOp.label(sIdxPrev) == 0 ) {
            err -= fOp.weight(sIdxPrev);
            errInv += fOp.weight(sIdxPrev);
        }

#else
        errInv = computeError( sampleIdxs, sampleClass, mat, column, weights, thr, true );
        err = computeError( sampleIdxs, sampleClass, mat, column, weights, thr, false );
#endif

        if ( (thr == prevThr) )
            continue;

        uniqueThresholds.push_back( thr );

        if (!TForcePolarity)
        {
            // normal
            if ( err < minErr ) {
                bestThr = thr;
                minErr = err;
                bestIdx = uniqueThresholds.size() - 1;
                bestInv = false;
            }

            if ( errInv < minErr ) {
                bestThr = thr;
                minErr = errInv;
                bestIdx = uniqueThresholds.size() - 1;
                bestInv = true;
            }
        }
        else
        {
            // polarity forced
            if (!inv)
            {
                if ( err < minErr ) {
                    bestThr = thr;
                    minErr = err;
                    bestIdx = uniqueThresholds.size() - 1;
                    bestInv = false;
                }
            }
            else
            {
                if ( errInv < minErr ) {
                    bestThr = thr;
                    minErr = errInv;
                    bestIdx = uniqueThresholds.size() - 1;
                    bestInv = true;
                }
            }
        }

    }

    if ( bestIdx  == 0 )
        retThr = uniqueThresholds[bestIdx];
    else
        retThr = ( uniqueThresholds[bestIdx] + uniqueThresholds[bestIdx-1] )/2;
    inv = bestInv;
    
    // compare errors

    if (false)
    {
        AdaBoostErrorType goodErr = computeError( fOp, retThr, inv );
        if (abs(minErr - goodErr) > 0.001)
                qDebug("Err: %.2f -> %f / %f", abs(minErr - goodErr), minErr, goodErr);
        //qDebug("  Thr: %f %f", bestThr, retThr );
    }


#if 0
    if ( column == 1 ) {
        qDebug("thr: %f %d %d %d %f", thr, inv, bestIdx, uniqueThresholds.size(), minErr);
        for (unsigned int i=0; i < uniqueThresholds.size(); i++)
            qDebug("T %d: %f", i, uniqueThresholds[i]);
    }
#endif
    minErr = computeError( fOp, retThr, inv );

    return minErr;
}

class HistogramMeanThreshold
{
public:
    typedef HistogramMeanThresholdData ParamType;
    typedef unsigned int     SampleIdxType;
    typedef std::vector<unsigned int>           SampleClassVector;
    typedef Eigen::ArrayXd  WeightsType;
    
    #if USE_DIFFERENCE_SPLIT
		typedef ThresholdSplitSubtract	SplitType;
    #else
		typedef ThresholdSplit	SplitType;
    #endif

private:
    const ParamType &mParams;

public:

    const ParamType & params() const {
        return mParams;
    }


    // trick from http://cplusplus.co.il/2009/09/04/implementing-assignment-operator-using-copy-constructor/
    HistogramMeanThreshold &operator= (const HistogramMeanThreshold &a)
    {
        if (this != &a)
        {
            this->HistogramMeanThreshold::~HistogramMeanThreshold();
            new (this) HistogramMeanThreshold(a);
        }
        return *this;
    }

    HistogramMeanThreshold( ) : mParams( *((ParamType *)0) ) {}

    HistogramMeanThreshold( const ParamType& param ) : mParams(param) {}

    // returns error
    // if prevSplit != null => works by refining the threshold, everything else is kept the same
    IntegralImageType learn( const std::vector<SampleIdxType> &sampleIdxs, const std::vector<unsigned int> &sampleClass, const WeightsType &weights, SplitType &tSplit,
                             const ThresholdSplit *prevSplit = 0 )
    {
                typedef FeatureRawOperator<HistogramMeanThresholdData, WeightsType>  		RawOperatorType;
                //typedef FeatureSubtractorOperator<HistogramMeanThresholdData, WeightsType>  SubtractorOperatorType;
		
        const bool refineThreshold = prevSplit != 0;

        qDebug("Total number of pose Idxs: %d\n", (int)mParams.numPosIdx());

        //const unsigned int N = sampleIdxs.size();
        const unsigned int numII = mParams.pixIntImages.size();

        AdaBoostErrorType minErr = 1e6;
        unsigned int bestBin = 0;
        unsigned bestPoseIdx = 0;
        bool bestInvert = false;
        IntegralImageType bestThr = 0;
        
        #if USE_DIFFERENCE_SPLIT
                unsigned int bestPoseIdxB = 0;
        #endif


        unsigned numExploredPoses = 0;
        unsigned numExploredWeakLearners = 0;

		//#warning learning 2k instead of 4k
        const unsigned numTotalWeakLearnersToExplore = mParams.numWLToExplore;
        const unsigned totalCombinations = numII * mParams.numPosIdx();

        const double combinationDivisor = sqrt(totalCombinations / numTotalWeakLearnersToExplore);

        unsigned numIIToExplore = numII;
        unsigned numPosesToExplore = std::min( (unsigned) ceil( numTotalWeakLearnersToExplore / numII ), (unsigned) mParams.numPosIdx() );

        std::vector<unsigned> poseIdxs( mParams.numPosIdx() );
        for (unsigned i=0; i < poseIdxs.size(); i++)
            poseIdxs[i] = i;

        std::random_shuffle( poseIdxs.begin(), poseIdxs.end() );

        unsigned int progressStep = numPosesToExplore / 200;
        if (progressStep < 2)   progressStep = 2;

        if (refineThreshold) {
            numPosesToExplore = 1;
            numIIToExplore = 1;

            poseIdxs[0] = prevSplit->poseIdx();
        }

        /**
          To make randomness controllable, let's prepare the bin idxs to search for for every pose idx
         **/

        std::vector< std::vector<unsigned> > binIdxsPerPose( numPosesToExplore );

        for (unsigned pi=0; pi < numPosesToExplore; pi++)
        {
            std::vector< unsigned > &binIdxs = binIdxsPerPose[pi];

            binIdxs.resize( numII );

            for (unsigned i=0; i < binIdxs.size(); i++)
                binIdxs[i] = i;

            std::random_shuffle( binIdxs.begin(), binIdxs.end() );
        }

        // for each possible pose
        #pragma omp parallel for schedule(dynamic)
        for (unsigned iPoseIdx = 0; iPoseIdx < numPosesToExplore; iPoseIdx++ )
        //for (unsigned iPoseIdx = 0; iPoseIdx < mParams.possibleOffsets.cols(); iPoseIdx++ )
        {
            // show some progress
            if ( (iPoseIdx % progressStep) == 1 )
            {
                printf("Progress: %.1f%%\r", iPoseIdx * 100.0 / numPosesToExplore);
                fflush(stdout);
            }

            const unsigned poseIdx = poseIdxs[iPoseIdx];
			            
        #if USE_DIFFERENCE_SPLIT
			for (unsigned poseIdxB = 0; poseIdxB < mParams.possibleOffsets.cols(); poseIdxB++ )
			{
				if (poseIdxB == poseIdx)	continue;
				
                if (rand() > RAND_MAX/60)
                    continue;
				
		#else
			{
		#endif

                            numExploredPoses++;

                            std::vector< unsigned > &binIdxs = binIdxsPerPose[iPoseIdx];

                            if (refineThreshold) {
                                qDebug("Refine thr");
                                binIdxs[0] = prevSplit->column();
                            }

                for (unsigned int iBin=0; iBin < numIIToExplore; iBin++)
                //for (unsigned int iBin=0; iBin < numII; iBin++)
				{
                    bool inv; IntegralImageType thr;
                    const unsigned bin = binIdxs[iBin];
					
					#if USE_DIFFERENCE_SPLIT
						AdaBoostErrorType err = findBestThreshold( 
                                                        SubtractorOperatorType( poseIdx, poseIdxB, weights, sampleClass, bin, mParams ), thr, inv );
					#else
                            AdaBoostErrorType err = 0;
                            if (refineThreshold)
                            {
                                inv = prevSplit->invert();
                                err = findBestThreshold<RawOperatorType, true>( RawOperatorType( poseIdx, weights, sampleClass, bin, mParams, sampleIdxs ), thr, inv );
                            }
                            else
                                err = findBestThreshold<RawOperatorType, false>( RawOperatorType( poseIdx, weights, sampleClass, bin, mParams, sampleIdxs ), thr, inv );
					#endif
					//AdaBoostErrorType err = findBestThreshold( sampleIdxs, sampleClass, mParams.svHist, bin, weights, thr, inv );

                                        #pragma omp critical
                                        {
                                            numExploredWeakLearners++;
                                            if ( err < minErr )
                                            {
                                                    minErr = err;
                                                    bestInvert = inv;
                                                    bestPoseIdx = poseIdx;
                                                    #if USE_DIFFERENCE_SPLIT
                                                            bestPoseIdxB = poseIdxB;
                                                    #endif
                                                    bestBin = bin;
                                                    bestThr = thr;
                                            }
                                        }
				}
			}
        }

		#if USE_DIFFERENCE_SPLIT
                        char cc[1024];
                        sprintf(cc, "Weak learner: bin %d, thr %.2f, err %.2f, p: %d, dA: %.2f, thA: %.1f, phA: %.1f, rA: %d dB: %.2f, thB: %.1f, phB: %.1f, rB: %d inv: %d", bestBin, bestThr, minErr,
				   bestPoseIdx, 
				   mParams.poseIndexMagnitude(bestPoseIdx),
				   mParams.poseIndexTheta(bestPoseIdx),
				   mParams.poseIndexPhi(bestPoseIdx),
                                   (int)mParams.poseIndexRadius(bestPoseIdx),
				   mParams.poseIndexMagnitude(bestPoseIdxB),
				   mParams.poseIndexTheta(bestPoseIdxB),
				   mParams.poseIndexPhi(bestPoseIdxB),
                                   (int)mParams.poseIndexRadius(bestPoseIdxB),
				    (int)bestInvert);

			tSplit = ThresholdSplitSubtract( bestBin, bestThr, bestInvert, bestPoseIdx, bestPoseIdxB );
                        tSplit.setStringDescrption( std::string(cc) );
		#else
                        char cc[1024];
                        //qDebug("Best pose idx: %d", (int)bestPoseIdx);
                        sprintf(cc, "Weak learner: bin %d, thr %.2f, err %.2f, p: %d, d: %.2f, th: %.1f, ph: %.1f r: %d inv: %d", bestBin, bestThr, minErr,
				   bestPoseIdx, 
				   mParams.poseIndexMagnitude(bestPoseIdx),
				   mParams.poseIndexTheta(bestPoseIdx),
                                   mParams.poseIndexPhi(bestPoseIdx),
                                (int) mParams.poseIndexRadius( bestPoseIdx ),
                                (int)bestInvert);
			tSplit = ThresholdSplit( bestBin, bestThr, bestInvert, bestPoseIdx );
                        tSplit.setStringDescrption( std::string(cc) );
		#endif

        qDebug("Explored poses: %d, weak learners: %d", (int)numExploredPoses, (int)numExploredWeakLearners );

        return minErr;
    }
};

#endif // HISTOGRAMMEANTHRESHOLD_H

