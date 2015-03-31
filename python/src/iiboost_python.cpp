#include <stdexcept>
#include <cstdio>
#include <cstring>

// error handling (name comes from Qt)
#define qFatal(...) do { char errStr[8192]; sprintf(errStr, "ERROR: "); sprintf(errStr, __VA_ARGS__);sprintf(errStr, "\n"); \
                          throw std::runtime_error(errStr); } while(0)
#define qDebug(...) do { fprintf (stdout, __VA_ARGS__); fprintf(stdout, "\n"); fflush(stdout); } while(0)


#include "ROIData.h"
#include "BoosterInputData.h"
#include "ContextFeatures/ContextRelativePoses.h"

#include "Booster.h"
#include "utils/TimerRT.h"

#include <Python.h>
#include "SmartPtrs.h"

typedef float PredictionPixelType;

#if defined(_MSC_VER)
#   define DLL_EXPORT __declspec(dllexport)
#else
#   define DLL_EXPORT
#endif

extern "C"
{
    DLL_EXPORT
    void freeModel( void *modelPtr )
    {
        if (modelPtr == 0)    return;

        delete ((BoosterModel *)modelPtr);
    }

    // returns py string object
    DLL_EXPORT
    PyObject * serializeModel( void *modelPtr )
    {
        std::string str;
        ((BoosterModel *)modelPtr)->serializeToString( &str );

        return PyString_FromString( str.c_str() );
    }

    // create model from serialized string
    // returns 0 on error
    DLL_EXPORT
    void * deserializeModel( const char *modelString )
    {
        BoosterModel *model = new BoosterModel();

        if (!model->deserializeFromString( modelString ))
        {
            delete model;
            return 0;
        }

        return model;
    }


    // Prediction for a single ROI
    //  Accepts an arbitrary number of integral images/channels.
    //  Assumes that predPtr is already allocated, of same size as imgPtr
    //  Returns 0 if ok
    DLL_EXPORT
    int predictWithChannels( void *modelPtr, ImagePixelType *imgPtr,
                              void *eigVecImgPtr,
                              int width, int height, int depth,
                              IntegralImagePixelType **chImgPtr,
                              int numChannels, double zAnisotropyFactor,
                              int useEarlyStopping,
                              PredictionPixelType *predPtr )
    {
        Matrix3D<PredictionPixelType> predMatrix;
        predMatrix.fromSharedData( predPtr, width, height, depth );

        // create roi for image, no GT available
        ROIData roi;
        roi.init( imgPtr, 0, 0, 0, width, height, depth, zAnisotropyFactor, 0.0, (const ROIData::RotationMatrixType *) eigVecImgPtr );
        std::unique_ptr<ROIData::IntegralImageType[]> ii(new ROIData::IntegralImageType[numChannels]);  // TODO: remove

        for (int ch=0; ch < numChannels; ch++)
        {
           ii[ch].fromSharedData(chImgPtr[ch], width, height, depth);

           roi.addII( ii[ch].internalImage().data() );
        }

        MultipleROIData allROIs;
        allROIs.add( shared_ptr_nodelete(ROIData, &roi) );

        try
        {
            Booster adaboost;
            adaboost.setModel( *((BoosterModel *) modelPtr) );
            if(useEarlyStopping != 0)
                adaboost.predictWithFeatureOrdering<true>( allROIs, &predMatrix );
            else
                adaboost.predictWithFeatureOrdering<false>( allROIs, &predMatrix );
        }
        catch( std::exception &e )
        {
            printf("Error in prediction: %s\n", e.what());
            return -1;
        }

        return 0;
    }

    // input: multiple imgPtr, gtPtr (arrays of pointers)
    //          multiple img sizes
    // returns a BoosterModel *
    // -- BEWARE: this function is a mix of dirty tricks right now
    DLL_EXPORT
    void * trainWithChannels( ImagePixelType **imgPtr, 
                             void **evecPtr,
                             GTPixelType **gtPtr,
                             int *width, int *height, int *depth,
                             int numStacks,
                             IntegralImagePixelType **chImgPtr,
                             int numChannels, double zAnisotropyFactor,
                             int numStumps,
                             int gtNegativeLabel, int gtPositiveLabel,
                             int debugOutput )
    {

        BoosterModel *modelPtr = 0;

        try
        {
            std::unique_ptr<ROIData[]> rois(new ROIData[numStacks]);  // TODO: remove
            MultipleROIData allROIs;
            std::unique_ptr<ROIData::IntegralImageType[]> ii(new ROIData::IntegralImageType[numStacks*numChannels]);  // TODO: remove

            for (int i=0; i < numStacks; i++)
            {
                rois[i].setGTNegativeSampleLabel(gtNegativeLabel);
                rois[i].setGTPositiveSampleLabel(gtPositiveLabel);
                rois[i].init( imgPtr[i], gtPtr[i], 0, 0, width[i], height[i], depth[i], 
                              zAnisotropyFactor, 0.0, (const ROIData::RotationMatrixType *) evecPtr[i] );

                for (int ch=0; ch < numChannels; ch++)
                {
                   ii[i*numChannels+ch].fromSharedData(chImgPtr[i*numChannels+ch], width[i], height[i], depth[i]);

                   rois[i].addII( ii[i*numChannels+ch].internalImage().data() );
                }

                allROIs.add( shared_ptr_nodelete(ROIData, &rois[i]) );
            }

            BoosterInputData bdata;
            bdata.init( shared_ptr_nodelete(MultipleROIData, &allROIs) );
            bdata.showInfo();

            Booster adaboost;
            adaboost.setShowDebugInfo( debugOutput != 0 );

            adaboost.train( bdata, numStumps );

            // create by copying
            modelPtr = new BoosterModel( adaboost.model() );
        }
        catch( std::exception &e )
        {
            printf("Error training: %s\n", e.what());
            delete modelPtr;

            return 0;
        }

        return modelPtr;
    }

    // input: one imgPtr (float32)
    // returns an IntegralImage< IntegralImagePixelType = float > image
    DLL_EXPORT 
    void computeIntegralImage( IntegralImagePixelType *rawImgPtr,
                               int width, int height, int depth,
                               IntegralImagePixelType *integralImagePtr)
    {

        Matrix3D<IntegralImagePixelType> integralMatrix;
        integralMatrix.fromSharedData( integralImagePtr, width, height, depth );

        Matrix3D<IntegralImagePixelType> rawImageMatrix;
        rawImageMatrix.fromSharedData( rawImgPtr, width, height, depth );

        IntegralImage<IntegralImagePixelType>::staticCompute( integralMatrix, rawImageMatrix );

    }


    /*** BEGIN OLD FUNCTIONALITY, IF NEEDED ***/

    // input: multiple imgPtr, gtPtr (arrays of pointers)
    //        multiple img sizes
    // returns a BoosterModel *
    // -- BEWARE: this function is a mix of dirty tricks right now
    DLL_EXPORT
    void * train( ImagePixelType **imgPtr, GTPixelType **gtPtr, 
                  int *width, int *height, int *depth,
                  int numStacks,
                  int numStumps, int debugOutput )
    {
        BoosterModel *modelPtr = 0;

        try
        {
            std::unique_ptr<ROIData[]> rois(new ROIData[numStacks]);  // TODO: remove
            std::unique_ptr<ROIData::IntegralImageType[]> ii(new ROIData::IntegralImageType[numStacks]);  // TODO: remove
            MultipleROIData allROIs;

            for (int i=0; i < numStacks; i++)
            {
                rois[i].init( imgPtr[i], gtPtr[i], 0, 0, width[i], height[i], depth[i] );

                // raw image to integral image
                // TODO: this should be removed and passed directly to train()
                ii[i].compute( rois[i].rawImage );
                rois[i].addII( ii[i].internalImage().data() );

                allROIs.add( shared_ptr_nodelete(ROIData, &rois[i]) );
            }

            BoosterInputData bdata;
            bdata.init( shared_ptr_nodelete(MultipleROIData, &allROIs) );
            bdata.showInfo();

            Booster adaboost;
            adaboost.setShowDebugInfo( debugOutput != 0 );

            adaboost.train( bdata, numStumps );

            // create by copying
            modelPtr = new BoosterModel( adaboost.model() );
        }
        catch( std::exception &e )
        {
            printf("Error training: %s\n", e.what());
            delete modelPtr;
            
            return 0;
        }

        return modelPtr;
    }

    // input: multiple imgPtr, gtPtr (arrays of pointers)
    //        multiple img sizes
    // returns a BoosterModel *
    // -- BEWARE: this function is a mix of dirty tricks right now
    DLL_EXPORT
    void * trainWithChannel( ImagePixelType **imgPtr, GTPixelType **gtPtr,
                             IntegralImagePixelType **chImgPtr,
                              int *width, int *height, int *depth,
                              int numStacks,
                              int numStumps, int debugOutput )
    {
        BoosterModel *modelPtr = 0;
        try
        {
            std::unique_ptr<ROIData[]> rois(new ROIData[numStacks]);  // TODO: remove
            MultipleROIData allROIs;

            for (int i=0; i < numStacks; i++)
            {
                rois[i].init( imgPtr[i], gtPtr[i], 0, 0, width[i], height[i], depth[i] );

                ROIData::IntegralImageType ii;
                ii.fromSharedData(chImgPtr[i], width[i], height[i], depth[i]);
                rois[i].addII( ii.internalImage().data() );

                allROIs.add( shared_ptr_nodelete(ROIData, &rois[i]) );
            }

            BoosterInputData bdata;
            bdata.init( shared_ptr_nodelete(MultipleROIData, &allROIs) );
            bdata.showInfo();

            Booster adaboost;
            adaboost.setShowDebugInfo( debugOutput != 0 );

            adaboost.train( bdata, numStumps );

            // create by copying
            modelPtr = new BoosterModel( adaboost.model() );
        }
        catch( std::exception &e )
        {
            printf("Error training: %s\n", e.what());
            delete modelPtr;

            return 0;
        }

        return modelPtr;
    }

    // Prediction
    //  Internally computes integral image from raw image, thus limited functionality.
    //  Assumes that predPtr is already allocated, of same size as imgPtr.
    DLL_EXPORT
    void predict( void *modelPtr, ImagePixelType *imgPtr, int width, int height, int depth, PredictionPixelType *predPtr )
    {
        Matrix3D<PredictionPixelType> predMatrix;
        predMatrix.fromSharedData( predPtr, width, height, depth );

        // create roi for image, no GT available
        ROIData roi;
        roi.init( imgPtr, 0, 0, 0, width, height, depth );

        // raw image to integral image
        ROIData::IntegralImageType ii;
        ii.compute( roi.rawImage );
        roi.addII( ii.internalImage().data() );


        MultipleROIData allROIs;
        allROIs.add( shared_ptr_nodelete(ROIData, &roi) );

        Booster adaboost;
        adaboost.setModel( *((BoosterModel *) modelPtr) );

        adaboost.predict<false>( allROIs, &predMatrix );
    }


    // Prediction
    //  Accepts a single integral image/channel, thus only to be used for testing
    //  Assumes that predPtr is already allocated, of same size as imgPtr
    DLL_EXPORT
    void predictWithChannel( void *modelPtr, ImagePixelType *imgPtr,
                              IntegralImagePixelType *chImgPtr,
                              int width, int height, int depth,
                              PredictionPixelType *predPtr )
    {
        Matrix3D<PredictionPixelType> predMatrix;
        predMatrix.fromSharedData( predPtr, width, height, depth );

        // create roi for image, no GT available
        ROIData roi;
        roi.init( imgPtr, 0, 0, 0, width, height, depth );

        // get the precomputed integral images
        ROIData::IntegralImageType ii;
        ii.fromSharedData(chImgPtr, width, height, depth);
        roi.addII( ii.internalImage().data() );

        MultipleROIData allROIs;
        allROIs.add( shared_ptr_nodelete(ROIData, &roi) );

        Booster adaboost;
        adaboost.setModel( *((BoosterModel *) modelPtr) );

        adaboost.predict<false>( allROIs, &predMatrix );
    }


    /*** Eigenvectors of Image wrappers ****/
    DLL_EXPORT
    void *computeEigenVectorsOfHessianImage( ImagePixelType *imgPtr, 
                                      int width, int height, int depth,
                                      double zAnisotropyFactor,
                                      double sigma )
    {
        Matrix3D<ImagePixelType> rawImg;
        rawImg.fromSharedData( imgPtr, width, height, depth );

        // compute eigen stuff
        ROIData::ItkEigenVectorImageType::Pointer rotImg = 
                AllEigenVectorsOfHessian::allEigenVectorsOfHessian<ImagePixelType>( 
                    sigma, zAnisotropyFactor, rawImg.asItkImage(), 
                    AllEigenVectorsOfHessian::EByMagnitude );

        rotImg->GetPixelContainer()->SetContainerManageMemory(false);

        return rotImg->GetPixelContainer()->GetImportPointer();
    }

    DLL_EXPORT
    void freeEigenVectorsOfHessianImage( void *ptr )
    {
        if ( ptr == 0 )
            return;
        
        typedef ROIData::ItkEigenVectorImageType::PixelType EigenVectorMatrixType;
        delete[] ((EigenVectorMatrixType *) ptr);
    }
}
