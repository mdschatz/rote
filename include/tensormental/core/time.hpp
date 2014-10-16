#ifndef _TIME_HPP_
#define _TIME_HPP_

#include <string>
#include <vector>
#include <list>
#include <mpi.h>
#include <time.h>

#ifdef PROFILE

#define PROFILE_SECTION(name) \
{ \
static Timer& __timer = Timer::timer(name); \
__timer.start();

#define PROFILE_FUNCTION PROFILE_SECTION(__func__)

#define PROFILE_STOP \
__timer.stop(); \
}

#define PROFILE_RETURN \
{ \
__timer.stop(); \
return; \
}

#define PROFILE_FLOPS(n) do_flops(n)

#else

#define PROFILE_FUNCTION

#define PROFILE_SECTION(name)

#define PROFILE_STOP

#define PROFILE_RETURN return;

#define PROFILE_FLOPS(n) do_flops(n)

#endif

class Timer;

class Interval
{
    friend class Timer;
    friend void do_flops(int64_t flops);
    friend Interval toc();
    friend Interval cputoc();

    protected:
        double dt;
        int64_t flops;

        Interval(double start, int64_t flops) : dt(start), flops(flops) {}

    public:
        Interval() : dt(0), flops(0) {}

        static Interval time();

        static Interval cputime();

        bool operator<(const Interval& other) const;

        Interval& operator+=(const Interval& other);

        Interval& operator-=(const Interval& other);

        Interval& operator*=(int m);

        Interval& operator/=(int m);

        Interval operator+(const Interval& other) const;

        Interval operator-(const Interval& other) const;

        Interval operator*(int m) const;

        Interval operator/(int m) const;

        double seconds() const;

        double gflops() const;
};

void tic();

Interval toc();

void cputic();

Interval cputoc();

void do_flops(int64_t flops);

class Timer
{
    protected:
        static std::list<Timer> timers;

        std::string name;
        Interval interval;
        int64_t count;

    public:
        Timer(const std::string& name) : name(name), interval(), count(0) {}

        bool operator<(const Timer& other) const;

        static Timer& timer(const std::string& name);

        void start() { tic(); }

        void stop()
        {
            #ifdef _OPENMP
            if (!omp_in_parallel())
            {
                interval += toc()*omp_get_max_threads();
                count++;
            }
            else
            #pragma omp critical
            #endif
            {
                interval += toc();
                count++;
            }
        }

        static void printTimers();

        static void clearTimers();

        double seconds() const { return interval.seconds(); }

        double gflops() const { return interval.gflops(); }
};

#endif
