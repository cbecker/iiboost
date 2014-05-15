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

#ifndef _ROI_DATA_H_
#define _ROI_DATA_H_

#include <Eigen/Dense>
#include <vector>

#include "globaldefs.h"

#include <Matrix3D.h>
#include "IntegralImage.h"

#if USE_SUPERVOXELS
    #include <slic/SuperVoxeler.h>
#endif

#include "auxItk/AllEigenVectorsOfHessian.h"

/**
 * Holds information for each Region of Interest, namely:
 *      - Rotation matrices
 *      - Raw image
 *      - Ground truth image
 *      - Integral images / channels
 *      - Mean-variance normalization (if USE_MEANVAR_NORMALIZATION != 0)
 */
struct ROIData
{
private:
    // rotation matrices, as Itk matrices, one per channel
    typedef AllEigenVectorsOfHessian::EigenVectorImageType  ItkEigenVectorImageType;
    ItkEigenVectorImageType::Pointer  rotMatricesImg;

public:

    // type of channels, needed specially for mean-var norm
    enum class ChannelType : char { GAUSS,       // smoothed gradient / image itself
                                    GRADIENT,    // gradient / gradient magnitude
                                    HESS_EIGVAL, // hessian eigenvalues
                                    STENS_EIGVAL, // structure tensor eigenvalues
                                    OTHER         // only working when not using mean-var normalization
                                  };

    // saves II type for each added II
    std::vector< ChannelType >          integralImageType;


#if USE_SUPERVOXELS
    SuperVoxeler<ImagePixelType>     SVox;
    std::vector<UIntPoint3D>         svCentroids;
#endif

    typedef IntegralImage<IntegralImagePixelType> IntegralImageType;

    // each integral image channel
    std::vector< IntegralImageType > integralImages;

    // for now we need this to compute rotMatrices
    Matrix3D<ImagePixelType>  rawImage;

    // ground truth
    // see GTIgnoreLabel, etc in globaldefs.h
    Matrix3D<GTPixelType>     gtImage;


    // ITK stores matrices in row major, but the eigenvectors are in the rows
    // of the vnl matrix
    typedef Eigen::Matrix<float,3,3,Eigen::ColMajor>     RotationMatrixType;

    // this will point to rotMatricesImg's data
    const RotationMatrixType *rotMatrices;

#if USE_MEANVAR_NORMALIZATION
    //  current implementation is very memory-intensive!!

    // this is for the raw image, so we can re-use it. one element per voxel
    std::vector<IntegralImagePixelType> rawImgMean;  //  mean around every voxel
    std::vector<IntegralImagePixelType> rawImgInvStdDev;  // 1 / std dev

    // one per integral image, containing one element for each voxel
    std::vector< std::vector<IntegralImagePixelType> >   meanVarMult;  //  to multiply by
    std::vector< std::vector<IntegralImagePixelType> >   meanVarSubtract;  // to subtract
#endif


//////////////////////////////////////////////////////////////////////////////////////

    // computes rotation matrices from raw image
    void updateRotationMatrices( const float rotHessianSigma, const float zAnisotropyFactor )
    {
        // compute rotation matrix
        rotMatricesImg = 
            AllEigenVectorsOfHessian::allEigenVectorsOfHessian<ImagePixelType>( 
                rotHessianSigma, zAnisotropyFactor, rawImage.asItkImage(), 
                AllEigenVectorsOfHessian::EByMagnitude );

        // convert pointers
        rotMatrices = (const RotationMatrixType *) rotMatricesImg->GetPixelContainer()->GetImportPointer();
    }

#if USE_MEANVAR_NORMALIZATION
    // compute meanVarMult and meanVarSubtract from data
    void addMeanVarianceNormParameters( ChannelType chanType )
    {
        // see if we need to compute raw-image stats
        if ( rawImgMean.size() != rawImage.numElem() )
        {
            // we need normal and squared II
            IntegralImageType rawII;
            IntegralImageType squaredII;
            {
                // temporary squared image
                Matrix3D<IntegralImagePixelType> squaredImg;
                squaredImg.reallocSizeLike( rawImage );

                for (unsigned i=0; i < rawImage.numElem(); i++)
                {
                    const IntegralImagePixelType val = rawImage.data()[i];
                    squaredImg.data()[i] = val*val;
                }

                // compute squared II
                squaredII.compute( squaredImg );
                rawII.compute( rawImage );
            }

            const int boxSize = MEANVAR_NORMALIZATION_CUBE_RADIUS;
            const int Vwidth = rawImage.width();
            const int Vheight = rawImage.height();
            const int Vdepth = rawImage.depth();

            rawImgMean.resize( rawImage.numElem() );
            rawImgInvStdDev.resize( rawImage.numElem() );

            #pragma omp parallel for
            for (unsigned i=0; i < rawImage.numElem(); i++)
            {
                unsigned _x,_y,_z;

                rawImage.idxToCoord( i, _x, _y, _z );
                int x = _x;
                int y = _y;
                int z = _z;

                // check image borders
                if ( x - boxSize <= 1 ) x = boxSize + 1;
                if ( y - boxSize <= 1 ) y = boxSize + 1;
                if ( z - boxSize <= 1 ) z = boxSize + 1;

                if ( x + boxSize >= Vwidth )   x = Vwidth - boxSize - 1;
                if ( y + boxSize >= Vheight)   y = Vheight - boxSize - 1;
                if ( z + boxSize >= Vdepth )   z = Vdepth - boxSize - 1;

                IntegralImagePixelType mean = rawII.centeredSumNormalized( x, y, z, boxSize, boxSize, boxSize, 0, 1 );

                IntegralImagePixelType stdDev = sqrt( squaredII.centeredSumNormalized( x, y, z, boxSize, boxSize, boxSize, 0, 1 ) - mean*mean );

        #if 0
                IntegralImageType realStdDev = 0;
                // compare std dev vs real one
                for (unsigned qx=x - boxSize; qx <= x + boxSize; qx++)
                    for (unsigned qy=y - boxSize; qy <= y + boxSize; qy++)
                        for (unsigned qz=z - boxSize; qz <= z + boxSize; qz++)
                        {
                            realStdDev += pow(combo.rawImage(qx,qy,qz) - mean, 2);
                        }

                double fR = 2*boxSize + 1;
                realStdDev = sqrt(realStdDev / (fR * fR * fR));
                qDebug("Real vs computed: %f %f", realStdDev, stdDev);

        #endif

                rawImgMean[i] = mean;
                rawImgInvStdDev[i] = 1.0 / (stdDev + 1e-6);
            }

        }

        // add channels
        meanVarMult.emplace_back();
        meanVarSubtract.emplace_back();

        switch( chanType )
        {
            case ChannelType::GAUSS:
                // mean-std dev not zero
                meanVarSubtract.back() = rawImgMean;
                meanVarMult.back() = rawImgInvStdDev;
                qDebug("Mean variance normalization Gauss");
                break;

            case ChannelType::GRADIENT:
                // mean zero, std dev
                meanVarSubtract.back().resize( rawImgMean.size() );
                for (auto &v: meanVarSubtract.back())   v = 0;

                meanVarMult.back() = rawImgInvStdDev;
                qDebug("Mean variance normalization Gradient");
                break;

            case ChannelType::HESS_EIGVAL:
                // mean zero, std dev
                meanVarSubtract.back().resize( rawImgMean.size() );
                for (auto &v: meanVarSubtract.back())   v = 0;

                meanVarMult.back() = rawImgInvStdDev;
                qDebug("Mean variance normalization Hess Eigval");
                break;

            case ChannelType::STENS_EIGVAL:
                // mean zero, std dev squared
                meanVarSubtract.back().resize( rawImgMean.size() );
                for (auto &v: meanVarSubtract.back())   v = 0;

                meanVarMult.back() = rawImgInvStdDev;
                for (auto &v: meanVarMult.back())   v = v * v;

                qDebug("Mean variance normalization STens Eigval");

                break;

            default:
                qFatal("Channel type of uknown type when using mean-variance normalization");
        }
    }
#endif

    // initialize from images, using move semantics
    void init( Matrix3D<ImagePixelType> &&rawImg, 
               Matrix3D<ImagePixelType> &&gtImg, 
               const float rotHessianSigma = 3.5,
               const float zAnisotropyFactor = 1.0 )
    {
        // in case we had other info before
        freeIntegralImages();

        rawImage = std::move(rawImg);
        gtImage = std::move(gtImg);

        updateRotationMatrices( rotHessianSigma, zAnisotropyFactor );
    }

    // initialize from pointers, which must remain valid
    void init( ImagePixelType *rawImgPtr, 
               ImagePixelType *gtImgPtr, 
               IntegralImagePixelType **intImgPtr,
               unsigned numII,
               unsigned width, unsigned height, unsigned depth,
               const float rotHessianSigma = 3.5,
               const float zAnisotropyFactor = 1.0 )
    {
#if USE_MEANVAR_NORMALIZATION
        qFatal("init() with pointers not supported with mean variance normalization, needs fix!");
#endif
        // in case we had other info before
        freeIntegralImages();

        rawImage.fromSharedData( rawImgPtr, width, height, depth );
        gtImage.fromSharedData( gtImgPtr, width, height, depth );

        updateRotationMatrices( rotHessianSigma, zAnisotropyFactor );

        for (unsigned i=0; i < numII; i++)
        {
            integralImages.push_back( IntegralImageType() );
            integralImages.back().fromSharedData( intImgPtr[i],
                                                   width, 
                                                   height,
                                                   depth );
        }
    }

    // it won't free pointer data
    void addII( IntegralImagePixelType *iiDataPtr, ChannelType chanType = ChannelType::OTHER )
    {
        if ( rawImage.isEmpty() )
            qFatal("Trued to add integral image to non-initialized ROIData");

        #if USE_MEANVAR_NORMALIZATION
            if ( chanType == ChannelType::OTHER )
                qFatal("Channel type cannot be OTHER with mean-var normalization");
        #endif

        integralImages.push_back( IntegralImageType() );
        integralImages.back().fromSharedData( iiDataPtr,
                                               rawImage.width(),
                                               rawImage.height(),
                                               rawImage.depth() );

        integralImageType.push_back( chanType );

        #if USE_MEANVAR_NORMALIZATION
            addMeanVarianceNormParameters( chanType );
        #endif
    }

    // to be used with std::move(), we love move semantics
    void addII( IntegralImageType &&iiDataPtr, ChannelType chanType = ChannelType::OTHER )
    {
        if ( rawImage.isEmpty() )
            qFatal("Trued to add integral image to non-initialized ROIData");

        #if USE_MEANVAR_NORMALIZATION
            if ( chanType == ChannelType::OTHER )
                qFatal("Channel type cannot be OTHER with mean-var normalization");
        #endif

        integralImages.push_back( std::move(iiDataPtr) );
        integralImageType.push_back( chanType );

        #if USE_MEANVAR_NORMALIZATION
            addMeanVarianceNormParameters( chanType );
        #endif
    }

    void freeIntegralImages()
    {
        integralImages.clear();
        #if USE_MEANVAR_NORMALIZATION
            meanVarMult.clear();
            meanVarSubtract.clear();
            integralImageType.clear();
        #endif
    }

    ~ROIData() 
    {
        freeIntegralImages();
    }


    // for debugging only
    void setAllOrientationsToIdentity()
    {
        for ( unsigned i=0; i < rawImage.numElem(); i++ )
        {
            // un-const it first
            ((RotationMatrixType *)rotMatrices)[i].setIdentity();
        }
    }
    
};

#endif