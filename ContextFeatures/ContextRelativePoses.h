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

#ifndef _CONTEXT_RELATIVE_POSES_H_
#define _CONTEXT_RELATIVE_POSES_H_

/** 
 * Possible location/radius of boxes
 * Pre-computes such list when constructed
 */

#include <Eigen/Dense>
#include "ROIData.h"
#include "BoosterInputData.h"
#include "IntegralImage.h"  // for BoxPosition


struct ContextRelativePoses
{
public:
    typedef Eigen::Vector3f LocType;    // TODO: should be shared with BoosterInputData

    // this type is just to make code easier to read
    struct FloatPoint3D
    {
        float x,y,z;
    };

	// this is what we need to access from outside
    std::vector<LocType>        possibleOffsets;			 // location relative to neutral coord system
    std::vector<unsigned int>   possibleOffsetsRadius;	     // radius, so that edge length = 2*r + 1

private:	// parameters, stay away!
	float  distMin, distMax;
    unsigned int distSteps;

    float  angThMin, angThMax;
    unsigned int angThSteps;
    
    float  angPhMin, angPhMax;
    unsigned int angPhSteps;

	unsigned int rMin;
    unsigned int rMax;
    unsigned int rSteps;

public:
    // returns number of pose idxs
    inline unsigned numPosIdx() const  { return possibleOffsets.size(); }

    // info about distance/angles
    float poseIndexMagnitude( unsigned poseIdx ) const
    {
        if (poseIdx >= possibleOffsets.size())  qFatal("Poseidx exceed possible value");
        return possibleOffsets[poseIdx].norm();
    }
    
    float poseIndexTheta( unsigned poseIdx ) const
    {
        if (poseIdx >= possibleOffsets.size())  qFatal("Poseidx exceed possible value");
        return 180.0 * acos( possibleOffsets[poseIdx](2) / possibleOffsets[poseIdx].norm() ) / M_PI;
    }
    
    float poseIndexPhi( unsigned poseIdx ) const
    {
        if (poseIdx >= possibleOffsets.size())  qFatal("Poseidx exceed possible value");
        return 180.0 * atan2( possibleOffsets[poseIdx](1), possibleOffsets[poseIdx](0) ) / M_PI;
    }

    unsigned poseIndexRadius( unsigned poseIdx ) const
    {
        if (poseIdx >= possibleOffsetsRadius.size())  qFatal("Poseidx exceed possible value");
        return possibleOffsetsRadius[poseIdx];
    }

    // given an image, the pose index and sample location
    inline void poseIndexedFeature( const BoosterInputData &BData,
                                    const unsigned poseIdx, const unsigned sampleIdx, 
                                    BoxPosition *box ) const
    {
        const ROIData &roi = *BData.imgData->ROIs[ BData.sampROI[sampleIdx] ];
        const float invZAnisotropyFactor = BData.imgData->invZAnisotropyFactor;

        const unsigned int Vwidth = roi.gtImage.width();
        const unsigned int Vheight = roi.gtImage.height();
        const unsigned int Vdepth = roi.gtImage.depth();

        const float radius = possibleOffsetsRadius[poseIdx];

        //for (unsigned int s=0; s < sampleIdxs.size(); s++)
        {
            const BoosterInputData::LocType &pt = BData.sampLocation[sampleIdx];

            // resize first
            Eigen::Vector3f orient = 
                    roi.rotMatrices[ BData.sampOffset[sampleIdx] ] * possibleOffsets[poseIdx];

            FloatPoint3D newPt;
            
            newPt.x = round(pt.coeff(0) + orient.coeff(0));
            newPt.y = round(pt.coeff(1) + orient.coeff(1));
            newPt.z = round(pt.coeff(2) + invZAnisotropyFactor * orient.coeff(2));

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

	ContextRelativePoses()
	{
		// TODO: make this more flexible, it depends on rStep in precomputeOrients()
		distMin = -20;
        distMax = 20;
        distSteps = 11;

        rMin = 0;
        rMax = 20;
        rSteps = 11;

        angThMin = 0;
        angThMax = M_PI/2 - 0.1;
        angThSteps =  6;

        angPhMin = 0.0;
        angPhMax = 2*M_PI;
        angPhSteps = 10;

        precomputeOrients();
	}

	void precomputeOrients()
    {
        possibleOffsetsRadius.clear();

        // TODO: link xthis with rSteps/rMin/rMax
		unsigned rStep = 2;

        possibleOffsets.resize( distSteps * angThSteps * angPhSteps * rSteps );
        possibleOffsetsRadius.resize( possibleOffsets.size() );

        double dStep = (distMax - distMin) / std::max( (distSteps - 1), 1U );
        double angThStep = (angThMax - angThMin) / std::max( (angThSteps - 1), 1U );
        double angPhStep = (angPhMax - angPhMin) / std::max( (angPhSteps - 1), 1U );

        // qDebug("dStep: %f", dStep);
        // qDebug("angThStep: %f", angThStep);
        // qDebug("angPhStep: %f", angPhStep);
        
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
				//std::cout << "Angvec: " << angleVec << std::endl;
				
				for (unsigned ds=0; ds < distSteps; ds++)
				{
					float d = distMin + dStep*ds;

                    for (unsigned r = rMin; r <= rMax; r += rStep)
                    {
                        possibleOffsets[ii] = d * angleVec;
                        possibleOffsetsRadius[ii] = r;

                        ii++;
                    }
				}
			}
		}

		qDebug("---> Number of pose indexes: %d (%d)", (int) possibleOffsets.size(), (int)ii );
            if (possibleOffsets.size() != ii)
                qFatal("error here, fix: %d %d!", (int)possibleOffsets.size(), (int)ii );
    }
};

#endif
