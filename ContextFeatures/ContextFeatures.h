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

#ifndef _CONTEXT_FEATURES_H_
#define _CONTEXT_FEATURES_H_

#include "IntegralImage.h"
#include <Eigen/Dense>


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

        {
            UIntPoint3D  pt = svCentroid[ sampleIdx ];

            // resize first
            Eigen::Vector3f orient = rotMatrices[sampleIdx] * possibleOffsets[poseIdx];
            FloatPoint3D newPt;
            
            newPt.x = round(pt.x + orient.coeff(0));
            newPt.y = round(pt.y + orient.coeff(1));
            newPt.z = round(pt.z + invZAnisotropyFactor * orient.coeff(2));

            //FIXME: shouldn't have to check for NaN
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

#endif