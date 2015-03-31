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

#ifndef _TIMER_RT_H_
#define _TIMER_RT_H_

#include <chrono>

class TimerRT
{
private:
    std::chrono::high_resolution_clock::time_point ts1;
public:
    void reset() {
        ts1 = std::chrono::high_resolution_clock::now();
    }

    TimerRT() { reset(); }

    double  elapsed() const
    {
        auto ts2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = ts2-ts1;
        return diff.count();
    }
};

#endif
