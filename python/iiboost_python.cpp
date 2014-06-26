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

typedef float PredictionPixelType;

extern "C"
{
	void freeModel( void *modelPtr )
	{
		if (modelPtr == 0)	return;

		delete ((BoosterModel *)modelPtr);
	}

	// returns py string object
	PyObject * serializeModel( void *modelPtr )
	{
		std::string str;
		((BoosterModel *)modelPtr)->serializeToString( &str );

		return PyString_FromString( str.c_str() );
	}

	// create model from serialized string
	// returns 0 on error
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

	// assumes that predPtr is already allocated, of same size as imgPtr
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
		allROIs.add( &roi );

		Booster adaboost;
		adaboost.setModel( *((BoosterModel *) modelPtr) );

		adaboost.predict( &allROIs, &predMatrix );
	}


    // assumes that predPtr is already allocated, of same size as imgPtr
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
        allROIs.add( &roi );

        Booster adaboost;
        adaboost.setModel( *((BoosterModel *) modelPtr) );

        adaboost.predict( &allROIs, &predMatrix );
    }

    // assumes that predPtr is already allocated, of same size as imgPtr
    void predictWithChannels( void *modelPtr, ImagePixelType *imgPtr,
                              int width, int height, int depth,
                              IntegralImagePixelType **chImgPtr,
                              int numChannels,
                              PredictionPixelType *predPtr )
    {
        Matrix3D<PredictionPixelType> predMatrix;
        predMatrix.fromSharedData( predPtr, width, height, depth );

        // create roi for image, no GT available
        ROIData roi;
        roi.init( imgPtr, 0, 0, 0, width, height, depth );

        // get the precomputed integral images
//        ROIData::IntegralImageType ii[numChannels];	// TODO: remove
//        for(int i = 0; i < numChannels; i++){
//            ii[i].fromSharedData(chImgPtr[i], width, height, depth);
//            //TODO what to do with it?
//            roi.addII( ii[i].internalImage().data() );
//        }
        //test with just one image //TODO remove this
        ROIData::IntegralImageType ii;
        ii.fromSharedData(chImgPtr[0], width, height, depth);
        roi.addII( ii.internalImage().data() );
        //

        MultipleROIData allROIs;
        allROIs.add( &roi );

        Booster adaboost;
        adaboost.setModel( *((BoosterModel *) modelPtr) );

        adaboost.predict( &allROIs, &predMatrix );
    }

	// input: multiple imgPtr, gtPtr (arrays of pointers)
	//		  multiple img sizes
	// returns a BoosterModel *
	// -- BEWARE: this function is a mix of dirty tricks right now
	void * train( ImagePixelType **imgPtr, GTPixelType **gtPtr, 
				  int *width, int *height, int *depth,
				  int numStacks,
				  int numStumps, int debugOutput )
	{
		BoosterModel *modelPtr = 0;

		try
		{
			ROIData rois[numStacks];					// TODO: not C++99 compatible?
			ROIData::IntegralImageType ii[numStacks];	// TODO: remove
			MultipleROIData allROIs;

			for (int i=0; i < numStacks; i++)
			{
				rois[i].init( imgPtr[i], gtPtr[i], 0, 0, width[i], height[i], depth[i] );

				// raw image to integral image
				// TODO: this should be removed and passed directly to train()
				ii[i].compute( rois[i].rawImage );
				rois[i].addII( ii[i].internalImage().data() );

				allROIs.add( &rois[i] );
			}

			BoosterInputData bdata;
			bdata.init( &allROIs );
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
    //		  multiple img sizes
    // returns a BoosterModel *
    // -- BEWARE: this function is a mix of dirty tricks right now
    void * trainWithChannel( ImagePixelType **imgPtr, GTPixelType **gtPtr,
                             IntegralImagePixelType **chImgPtr,
                              int *width, int *height, int *depth,
                              int numStacks,
                              int numStumps, int debugOutput )
    {
        BoosterModel *modelPtr = 0;
        try
        {
            ROIData rois[numStacks];					// TODO: not C++99 compatible?
            MultipleROIData allROIs;

            for (int i=0; i < numStacks; i++)
            {
                rois[i].init( imgPtr[i], gtPtr[i], 0, 0, width[i], height[i], depth[i] );

                ROIData::IntegralImageType ii;
                ii.fromSharedData(chImgPtr[i], width[i], height[i], depth[i]);
                rois[i].addII( ii.internalImage().data() );

                allROIs.add( &rois[i] );
            }

            BoosterInputData bdata;
            bdata.init( &allROIs );
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
    //		  multiple img sizes
    // returns a BoosterModel *
    // -- BEWARE: this function is a mix of dirty tricks right now
    void * trainWithChannels( ImagePixelType **imgPtr, GTPixelType **gtPtr,
                             int *width, int *height, int *depth,
                             int numStacks,
                             IntegralImagePixelType **chImgPtr,
                             int numChannels,
                             int numStumps, int debugOutput )
    {

        BoosterModel *modelPtr = 0;

        try
        {

            ROIData rois[numStacks];					// TODO: not C++99 compatible?
            MultipleROIData allROIs;
            ROIData::IntegralImageType ii[numStacks][numChannels];	// TODO: remove
//            ROIData::IntegralImageType testii[numStacks][numChannels];	// TODO: remove

            fprintf(stderr,"numchannels: %d numstacks: %d\n",numChannels,numStacks);

            for (int i=0; i < numStacks; i++)
            {
                for (int ch=0; ch < numChannels; ch++)
                {

                   rois[i].init( imgPtr[i], gtPtr[i], 0, 0, width[i], height[i], depth[i] );

//                   testii[i][ch].compute( rois[i].rawImage );
                   ii[i][ch].fromSharedData(chImgPtr[i*numChannels+ch], width[i], height[i], depth[i]);

                   rois[i].addII( ii[i][ch].internalImage().data() );

                   allROIs.add( &rois[i] );
                }
            }

            // test that both matrices are the same

//            for (int i=0; i < numStacks; i++)
//              for (int ch=0; ch < numChannels; ch++)
//                for(int w=0; w<width[i]; w++)
//                  for(int h=0; h<height[i]; h++)
//                    for(int d=0; d<depth[i]; d++)
//                      assert(     ii[i][ch].internalImage().data()[w*width[i]*height[i] +h*height[i] + d]
//                           == testii[i][ch].internalImage().data()[w*width[i]*height[i] +h*height[i] + d]);

            BoosterInputData bdata;
            bdata.init( &allROIs );
            bdata.showInfo();

            Booster adaboost;
            adaboost.setShowDebugInfo( debugOutput != 0 );

            adaboost.train( bdata, numStumps );

            // create by copying
            modelPtr = new BoosterModel( adaboost.model() );
        }
        catch( std::exception &e )
        {
            fprintf(stderr, "hey");

            printf("Error training: %s\n", e.what());
            delete modelPtr;

            return 0;
        }

        return modelPtr;
    }

    // input: one imgPtr (float32)
    // returns an IntegralImage< IntegralImagePixelType = float > image
    void computeIntegralImage( IntegralImagePixelType *rawImgPtr,
                               int width, int height, int depth,
                               IntegralImagePixelType *integralImagePtr){

        Matrix3D<IntegralImagePixelType> integralMatrix;
        integralMatrix.fromSharedData( integralImagePtr, width, height, depth );

        Matrix3D<IntegralImagePixelType> rawImageMatrix;
        rawImageMatrix.fromSharedData( rawImgPtr, width, height, depth );

        IntegralImage<IntegralImagePixelType>::staticCompute( integralMatrix, rawImageMatrix );

    }
}
