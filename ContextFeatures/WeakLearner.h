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

#ifndef _WEAK_LEARNER_
#define _WEAK_LEARNER_

#include <string>
#include "globaldefs.h"

// for serialization
#include <rapidjson/document.h>

/**
 * Weak learner class. Uses the poses from ContextRelativePoses.h
 *  and provides training and prediction.
 * Once trained, can be accessed through WeakLearner.
 */
class WeakLearner
{
private:
    IntegralImagePixelType mThreshold;
    bool  mInvert;
    unsigned int mChannel;
    unsigned int mPoseIdx;
    std::string mDescription;

public:

    WeakLearner() {}
    WeakLearner( unsigned int channel, IntegralImagePixelType threshold, bool invert, unsigned int poseIdx ) {
        mThreshold = threshold;
        mInvert = invert;
        mChannel = channel;
        mPoseIdx = poseIdx;
    }

    unsigned int poseIdx() const { return mPoseIdx; }

    // JSON serialization, requires passing the allocator bcos of rapidjson
    void serialize( rapidjson::Value &obj, 
                    rapidjson::Document::AllocatorType& allocator ) const
    {
        obj.SetObject();

        obj.AddMember("threshold", mThreshold, allocator);
        obj.AddMember("invert", mInvert, allocator);
        obj.AddMember("channel", mChannel, allocator);
        obj.AddMember("poseIdx", mPoseIdx, allocator);
        obj.AddMember("description", mDescription.c_str(), allocator);
    }

    // JSON deserialization
    bool deserialize( const rapidjson::Value &obj )
    {
        if (!obj.IsObject())
            return false;

        mThreshold = obj["threshold"].GetDouble();
        mInvert = obj["invert"].GetBool();
        mChannel = obj["channel"].GetInt();
        mPoseIdx = obj["poseIdx"].GetInt();
        mDescription = obj["description"].GetString();
    }

#ifdef LIBCONFIGXX_VER_MAJOR
    void save( libconfig::Setting &s ) const
    {
        s.add("threshold", libconfig::Setting::TypeFloat) = mThreshold;
        s.add("invert", libconfig::Setting::TypeBoolean) = mInvert;
        s.add("channel", libconfig::Setting::TypeInt) = (int)mChannel;
        s.add("poseIdx", libconfig::Setting::TypeInt) = (int)mPoseIdx;
        s.add("description", libconfig::Setting::TypeString) = mDescription;
    }

    void load( const libconfig::Setting &s )
    {
        mThreshold = s["threshold"];
        mInvert = s["invert"];
        mChannel = (int) s["channel"];
        mPoseIdx = (int) s["poseIdx"];
        mDescription = (const char *) s["description"];
    }
#endif

    IntegralImagePixelType threshold() const { return mThreshold; }
    bool invert() const { return mInvert; }
    unsigned int channel() const { return mChannel; }
    void invertClassifier() { mInvert = !mInvert; }
    
    void setInvert( bool yes ) { mInvert = yes; }
    void setThreshold( IntegralImagePixelType thr ) { mThreshold = thr; }

    inline void setStringDescription( const std::string &descr )
    {
        mDescription = descr;
    }

    inline const std::string &getStringDescription() const
    {
        return mDescription;
    }
    

    // template<typename SampleIdxVector, typename PredType>
    // void classify( const SampleIdxVector &sampleIdxs, 
    //                const ContextRelativePoses &poses,
    //                const MultipleROIData &imgData,
    //                std::vector<PredType> &prediction, const unsigned numThreads = 1 )
    // {
    //     prediction.resize( sampleIdxs.size() );
    //     classifyLowLevel<SampleIdxVector, PredType, 0>( sampleIdxs, poses, imgData, prediction.data(), numThreads );
    // }

    // template<typename SampleIdxVector, int negativeValue>
    // void classify( const SampleIdxVector &sampleIdxs, 
    //                const ContextRelativePoses &poses,
    //                const MultipleROIData &imgData,
    //                Eigen::ArrayXf &prediction, const unsigned numThreads = 1 )
    // {
    //     prediction.resize( sampleIdxs.size() );
    //     classifyLowLevel<SampleIdxVector, float, negativeValue>( sampleIdxs, poses, imgData, prediction.data(), numThreads );
    // }

    template<typename PredOpType>
    void classifySingleROIWithOp(
                           const ContextRelativePoses &poses,
                           const BoosterInputData &bid,
                           PredOpType &predOp,
                           const unsigned numThreads = 1) const
    {
        typedef typename ROIData::IntegralImageType IntegralImageType;
        const MultipleROIData &imgData = *bid.imgData;

        const unsigned roiNo = 0;

        const unsigned N = bid.sampROI.size();
        const IntegralImageType &ii = imgData.ROIs[roiNo]->integralImages[mChannel];

    #if USE_MEANVAR_NORMALIZATION
        const std::vector<IntegralImagePixelType> &mean = imgData.ROIs[roiNo]->meanVarSubtract[mChannel];
        const std::vector<IntegralImagePixelType> &invStd = imgData.ROIs[roiNo]->meanVarMult[mChannel];
    #endif

        if (mInvert == false)
        {
            #pragma omp parallel for num_threads(numThreads)
            for (unsigned int i=0; i < N; i++)
            {
                BoxPosition box;
                poses.poseIndexedFeature( bid, mPoseIdx, i, &box );
                
            #if USE_MEANVAR_NORMALIZATION
                if ( ii.centeredSumNormalized( box, mean[bid.sampOffset[i]], invStd[bid.sampOffset[i]] ) >= mThreshold )
            #else
                if ( ii.centeredSumNormalized( box ) >= mThreshold )
            #endif
                    predOp(i, true);
                else
                    predOp(i, false);
            }
        } else {
            #pragma omp parallel for num_threads(numThreads)
            for (unsigned int i=0; i < N; i++)
            {
                BoxPosition box;
                poses.poseIndexedFeature( bid, mPoseIdx, i, &box );
            
            #if USE_MEANVAR_NORMALIZATION
                if ( ii.centeredSumNormalized( box, mean[bid.sampOffset[i]], invStd[bid.sampOffset[i]] ) < mThreshold )
            #else
                if ( ii.centeredSumNormalized( box ) < mThreshold )
            #endif
                    predOp(i, true);
                else
                    predOp(i, false);
            }
        }
    }

    void classifyMultipleROIs(
                   const ContextRelativePoses &poses,
                   const BoosterInputData &bid,
                   Eigen::ArrayXf &prediction, const unsigned numThreads = 1 ) const
    {
        prediction.resize( bid.sampROI.size() );
        classifyMultipleROIsLowLevel<float, -1>( poses, bid, prediction.data(), numThreads );
    }

    template<typename PredType, int negativeValue>
    void classifyMultipleROIsLowLevel(
                           const ContextRelativePoses &poses,
                           const BoosterInputData &bid,
                           PredType *prediction, 
                           const unsigned numThreads = 1 ) const
    {
        typedef typename ROIData::IntegralImageType IntegralImageType;
        const MultipleROIData &imgData = *bid.imgData;

        const unsigned N = bid.sampROI.size();

        if (mInvert == false)
        {
            #pragma omp parallel for num_threads(numThreads)
            for (unsigned int i=0; i < N; i++)
            {
                BoxPosition box;
                poses.poseIndexedFeature( bid, mPoseIdx, i, &box );
                
                const unsigned roiNo = bid.sampROI[i];
                const IntegralImageType &ii = imgData.ROIs[roiNo]->integralImages[mChannel];

                #if USE_MEANVAR_NORMALIZATION
                    const std::vector<IntegralImagePixelType> &mean = imgData.ROIs[roiNo]->meanVarSubtract[mChannel];
                    const std::vector<IntegralImagePixelType> &invStd = imgData.ROIs[roiNo]->meanVarMult[mChannel];
                #endif


            #if USE_MEANVAR_NORMALIZATION
                if ( ii.centeredSumNormalized( box, mean[bid.sampOffset[i]], invStd[bid.sampOffset[i]] ) >= mThreshold )
            #else
                if ( ii.centeredSumNormalized( box ) >= mThreshold )
            #endif
                    prediction[i] = 1;
                else
                    prediction[i] = negativeValue;
            }
        } else {
            #pragma omp parallel for num_threads(numThreads)
            for (unsigned int i=0; i < N; i++)
            {
                BoxPosition box;
                poses.poseIndexedFeature( bid, mPoseIdx, i, &box );
                
                const unsigned roiNo = bid.sampROI[i];
                const IntegralImageType &ii = imgData.ROIs[roiNo]->integralImages[mChannel];

                #if USE_MEANVAR_NORMALIZATION
                    const std::vector<IntegralImagePixelType> &mean = imgData.ROIs[roiNo]->meanVarSubtract[mChannel];
                    const std::vector<IntegralImagePixelType> &invStd = imgData.ROIs[roiNo]->meanVarMult[mChannel];
                #endif

            #if USE_MEANVAR_NORMALIZATION
                if ( ii.centeredSumNormalized( box, mean[bid.sampOffset[i]], invStd[bid.sampOffset[i]] ) < mThreshold )
            #else
                if ( ii.centeredSumNormalized( box ) < mThreshold )
            #endif
                    prediction[i] = 1;
                else
                    prediction[i] = negativeValue;
            }
        }
    }

};


#include "BoosterInputData.h"


/** Computes the value of single feature for the set of samples requested **/
template<typename WeightsType, bool cacheValues = true>
struct FeatureRawOperator
{
    const ContextRelativePoses &mCCPoses;
    const unsigned mPoseIdx;
    const BoosterInputData &mBI;
    const WeightsType      &mWeights;
    const unsigned int mChannel;   // from which integral image to compute the features
    std::vector< IntegralImagePixelType > mCachedValues;
    
    inline FeatureRawOperator( const ContextRelativePoses &ccPoses,
                               const unsigned poseIdx, 
                               const BoosterInputData &bi,
                               const WeightsType &weights, 
                               unsigned int channel ) :
                    mCCPoses(ccPoses), mPoseIdx(poseIdx), mBI(bi), 
                    mWeights(weights),
                    mChannel(channel)
    {
        if (cacheValues)
        {
            mCachedValues.resize( mWeights.size() );

            for (unsigned i=0; i < mWeights.size(); i++)
                mCachedValues[i] = nonCachedValue(i);
        }
    }

    inline IntegralImagePixelType nonCachedValue( unsigned int idx ) const
    {
        BoxPosition box;
        
        mCCPoses.poseIndexedFeature( mBI, mPoseIdx, idx, &box );

        const ROIData &roi = *mBI.imgData->ROIs[ mBI.sampROI[idx] ];

    #if USE_MEANVAR_NORMALIZATION
        const std::vector<IntegralImagePixelType> &mean = roi.meanVarSubtract[mChannel];
        const std::vector<IntegralImagePixelType> &invStd = roi.meanVarMult[mChannel];

        return roi.integralImages[mChannel].centeredSumNormalized( box, mean[mBI.sampOffset[idx]], invStd[mBI.sampOffset[idx]] );
    #else
        return roi.integralImages[mChannel].centeredSumNormalized( box );
    #endif
    }

    inline IntegralImagePixelType value( unsigned int idx ) const
    {
        if (cacheValues)
            return mCachedValues[idx];
        else
            return nonCachedValue(idx);
    }
    
    inline typename WeightsType::Scalar weight( unsigned int idx ) const {
        return mWeights.coeff( idx );
    }
    
    inline GTPixelType label( unsigned int idx ) const {
        return mBI.sampLabels[idx];
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

template<typename FeatureOperatorType>
AdaBoostErrorType computeError(  const FeatureOperatorType &fOp,
                     IntegralImagePixelType threshold, bool inverted )
{
    AdaBoostErrorType err = 0;

    if (inverted == false)
    {
        for (unsigned int i=0; i < fOp.count(); i++)
        {
            if ( fOp.value(i) >= threshold ) {
                if ( fOp.label(i) == GTNegLabel )
                    err += fOp.weight(i);
            } else {
                if ( fOp.label(i) != GTNegLabel )
                    err += fOp.weight(i);
            }
        }
    } else {
        for (unsigned int i=0; i < fOp.count(); i++)
        {
            if ( fOp.value(i) < threshold ) {
                if ( fOp.label(i) == GTNegLabel )
                    err += fOp.weight(i);
            } else {
                if ( fOp.label(i) != GTNegLabel )
                    err += fOp.weight(i);
            }
        }
    }

    return err;
}

// if forcePolarity = true, then it will use the polarity found in 'inv'
template<typename FeatureOperatorType, bool TForcePolarity>
inline AdaBoostErrorType findBestThreshold( const FeatureOperatorType &fOp,
                        IntegralImagePixelType &retThr, bool &inv )
{
    const unsigned int N = fOp.count();
    
    std::vector<unsigned int> sortedIdx(N);  //alloc
    for (unsigned int i=0; i < N; i++)
        sortedIdx[i] = i;

    std::sort( sortedIdx.begin(), sortedIdx.end(),
               SortFeature<FeatureOperatorType>( fOp ) );

    // find out the error if we put threshold on zero
    AdaBoostErrorType minErr = 1e6;
    IntegralImagePixelType bestThr = 0;
    bool bestInv = false;
    unsigned int bestIdx = 0;


    // pre-compute errors with threshold on first element
    IntegralImagePixelType firstThr = fOp.value( sortedIdx[0] );
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

    std::vector<IntegralImagePixelType>  uniqueThresholds;
    uniqueThresholds.push_back( firstThr );

    for (unsigned int i=1; i < N; i++)
    {
        const unsigned int sIdxThis = sortedIdx[i];
        const unsigned int sIdxPrev = sortedIdx[i-1];

        IntegralImagePixelType thr = fOp.value( sIdxThis );
        IntegralImagePixelType prevThr = fOp.value( sIdxPrev );

        // not inv => changes if
        if ( fOp.label(sIdxPrev) == GTPosLabel ) {
            err += fOp.weight(sIdxPrev);
            errInv -= fOp.weight(sIdxPrev);
        }
        else if ( fOp.label(sIdxPrev) == GTNegLabel ) {
            err -= fOp.weight(sIdxPrev);
            errInv += fOp.weight(sIdxPrev);
        }


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
    
    minErr = computeError( fOp, retThr, inv );

    return minErr;
}


class WeakLearnerSearcher
{
public:
    typedef Eigen::ArrayXd  WeightsType;

private:
    const ContextRelativePoses &mPoses;
    const BoosterInputData     &mBID;
    const WeightsType          &mWeights;
    const unsigned             mNumWLToExplore;

public:

    WeakLearnerSearcher( const ContextRelativePoses &poses,
                         const BoosterInputData &bid,
                         const WeightsType &weights,
                         const unsigned numWLToExplore  ) :
        mPoses(poses), mBID(bid), mWeights(weights), mNumWLToExplore(numWLToExplore)
    {}


    // returns error
    // if prevSplit != null => works by refining the threshold, everything else is kept the same
    double learn( WeakLearner &tSplit, const bool showDebugInfo = false,
                  const WeakLearner *prevSplit = 0 ) const
    {
        typedef FeatureRawOperator<WeightsType>         RawOperatorType;
        
        const bool refineThreshold = prevSplit != 0;
        const unsigned numPosIdx = mPoses.numPosIdx();

        if (showDebugInfo)
            qDebug("Total number of pose Idxs: %d\n", (int)numPosIdx);

        const unsigned numII = mBID.imgData->ROIs[0]->integralImages.size();

        AdaBoostErrorType minErr = 1e6;
        unsigned int bestBin = 0;
        unsigned bestPoseIdx = 0;
        bool bestInvert = false;
        IntegralImagePixelType bestThr = 0;

        unsigned numExploredPoses = 0;
        unsigned numExploredWeakLearners = 0;

        //#warning learning 2k instead of 4k
        const unsigned numTotalWeakLearnersToExplore = mNumWLToExplore;
        const unsigned totalCombinations = numII * numPosIdx;

        const double combinationDivisor = sqrt(totalCombinations / numTotalWeakLearnersToExplore);

        unsigned numIIToExplore = numII;
        unsigned numPosesToExplore = std::min( (unsigned) ceil( numTotalWeakLearnersToExplore / numII ), (unsigned) numPosIdx );

        std::vector<unsigned> poseIdxs( numPosIdx );
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
        {
            // show some progress
            if (showDebugInfo)
                if ( (iPoseIdx % progressStep) == 1 )
                {
                    printf("Progress: %.1f%%\r", iPoseIdx * 100.0 / numPosesToExplore);
                    fflush(stdout);
                }

            const unsigned poseIdx = poseIdxs[iPoseIdx];
                        
            {
                numExploredPoses++;

                std::vector< unsigned > &binIdxs = binIdxsPerPose[iPoseIdx];

                if (refineThreshold) {
                    if (showDebugInfo)
                        qDebug("Refine thr");
                    binIdxs[0] = prevSplit->channel();
                }

                for (unsigned int iBin=0; iBin < numIIToExplore; iBin++)
                {
                    bool inv; IntegralImagePixelType thr;
                    const unsigned bin = binIdxs[iBin];
                    
                    AdaBoostErrorType err = 0;
                    if (refineThreshold)
                    {
                        inv = prevSplit->invert();
                        err = findBestThreshold<RawOperatorType, true>( RawOperatorType( 
                                                                            mPoses, poseIdx, 
                                                                            mBID, mWeights,
                                                                            bin ), thr, inv );
                    }
                    else
                        err = findBestThreshold<RawOperatorType, false>( RawOperatorType( 
                                                                            mPoses, poseIdx, 
                                                                            mBID, mWeights,
                                                                            bin ), thr, inv );
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

        // create weak learner
        char cc[1024];
        sprintf(cc, "Weak learner: bin %d, thr %.2f, err %.2f, p: %d, d: %.2f, th: %.1f, ph: %.1f r: %d inv: %d", bestBin, bestThr, minErr,
                   bestPoseIdx, 
                   mPoses.poseIndexMagnitude(bestPoseIdx),
                   mPoses.poseIndexTheta(bestPoseIdx),
                   mPoses.poseIndexPhi(bestPoseIdx),
                   (int) mPoses.poseIndexRadius( bestPoseIdx ),
                   (int)bestInvert);

        tSplit = WeakLearner( bestBin, bestThr, bestInvert, bestPoseIdx );
        tSplit.setStringDescription( std::string(cc) );

        if (showDebugInfo)
            qDebug("Explored poses: %d, weak learners: %d", (int)numExploredPoses, (int)numExploredWeakLearners );

        return minErr;
    }
};


#endif
