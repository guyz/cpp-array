/*
 * Copyright (C) 2011 by Alejandro M. Aragón
 * Written by Alejandro M. Aragón <alejandro.aragon@gmail.com>
 * All Rights Reserved
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

//! \file timer.hpp
//
//  Created by Alejandro Aragón on 10/18/11.
//  Copyright (c) 2011 University of Illinois at Urbana-Champaign. All rights reserved.
//

#ifndef ARRAY_TIMER_HPP
#define ARRAY_TIMER_HPP

#include <cmath>
#include <limits>
#include <ctime>
#include <sys/time.h>

namespace array {
    
	class time {
        
        typedef size_t time_unit;
        
        double seconds_;
        time_unit minutes_;
        time_unit hours_;
        time_unit days_;
        
    public:
        
        time(double s = 0, time_unit min = 0, time_unit h = 0, time_unit d = 0)
		: seconds_(s), minutes_(min), hours_(h), days_(d) {
            adjust();
        }
        
        time& operator+=(const time& t) {
            seconds_ += t.seconds_;
            minutes_ += t.minutes_;
            hours_ += t.hours_;
            days_ += t.days_;
            adjust();
        }
        
        //! Increment time
        /*! \param s - Time in seconds to be added to the time object.
         */
        time& operator+=(double s) {
            seconds_ += s;
            adjust();
            return *this;
        }
        
        time operator+(const time& t2) {
            time t(*this);
            t += t2;
            return t;
        }
        
        friend std::ostream& operator<<(std::ostream& os, const time& t) {
			if (t.days_ != 0)
				os<<t.days_<<" d ";
			if (t.hours_ != 0)
				os<<t.hours_<<" h ";
			if (t.minutes_ != 0)
				os<<t.minutes_<<" m ";
            os<<t.seconds_<<" s";
			return os;
        }
        
    private:
        
        void adjust() {
            if (seconds_ >= 60.) {
                minutes_ += static_cast<time_unit>(seconds_ / 60.);
                seconds_ = fmod(seconds_, 60.);
            }
            if (minutes_ >= 60) {
                hours_ += minutes_ / 60;
                minutes_ = minutes_ % 60;
            }
            if (hours_ >= 24) {
                days_ += hours_ / 24;
                hours_ = hours_ % 24;
            }
        }
    };
	
    
	///////////////////////////////////////////////////////////////////////////////
	// timer class
	
	class timer {
		
        std::clock_t t1_;
		
	public:
		timer() : t1_(std::clock()) {}
        
		
		double tac() const {
            return static_cast<double>(std::clock() - t1_)/CLOCKS_PER_SEC;
		}
        
        double max() const {
            return static_cast<double>(std::numeric_limits<std::clock_t>::max()
                                       - t1_) / CLOCKS_PER_SEC;
        }
        
        double min() const
        { return 1./CLOCKS_PER_SEC; }
        
		inline void reset() { t1_ = std::clock(); }
		
		friend std::ostream& operator<<(std::ostream& os, const timer& timer) {
			time t(timer.tac());
			os<<t;
			return os;
        }
	};
    
	class ctimer {
		
		double t1_;
		
	public:
		ctimer() {
			reset();
		}
		
		double tac() const {
			timeval end;
			gettimeofday(&end, NULL);
			double t2 = end.tv_sec + end.tv_usec/1000000.0;
			return t2 - t1_;
		}
        
		inline void reset() {
			timeval start;
			gettimeofday(&start, NULL);
			// Convert time format to seconds.microseconds double
			t1_ = start.tv_sec + start.tv_usec/1000000.0;
		}
		
		friend std::ostream& operator<<(std::ostream& os, const ctimer& timer) {
			time t(timer.tac());
			os<<t;
			return os;
        }
		
	};
    
	
} /* namespace array */

#endif /* ARRAY_TIMER_HPP */
