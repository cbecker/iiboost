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

#include <time.h>

#include <time.h>
#include <sys/time.h>

#ifdef __MACH__
    #include <mach/clock.h>
    #include <mach/mach.h>
#endif

// http://stackoverflow.com/a/6725161/162094
void gettime(timespec & ts) {
    #ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
        clock_serv_t cclock;
        mach_timespec_t mts;
        host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
        clock_get_time(cclock, &mts);
        mach_port_deallocate(mach_task_self(), cclock);
        ts.tv_sec = mts.tv_sec;
        ts.tv_nsec = mts.tv_nsec;
    #else
        clock_gettime(CLOCK_REALTIME, &ts);
    #endif
}

// a simple timer
class TimerRT
{
private:
    struct timespec ts1;
public:
    void reset() {
        gettime(ts1);
    }

    TimerRT() { reset(); }

    double  elapsed() const
    {
        struct timespec ts2;
        gettime(ts2);

        return (double) ( 1.0*(1.0*ts2.tv_nsec - ts1.tv_nsec*1.0)*1e-9
                         + 1.0*ts2.tv_sec - 1.0*ts1.tv_sec );
    }
};

#endif
