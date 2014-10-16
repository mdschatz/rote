#include "tensormental/core/time.hpp"
#ifdef __MACH__
#include <mach/mach_time.h>
#endif

using namespace std;

static std::vector<Interval> *tics[128];

Interval Interval::time()
{
    #ifdef _OPENMP
    int tid = omp_get_thread_num();
    #else
    int tid = 0;
    #endif

    #ifdef __MACH__
    static double conv = -1.0;
    if (conv < 0)
    {
        mach_timebase_info_data_t timebase;
        mach_timebase_info(&timebase);
        conv = (double)timebase.numer / (double)timebase.denom;
    }
    uint64_t nsec = mach_absolute_time();
    return Interval(conv*(double)nsec/1e9, 0);
    #else
    timespec ts;
//    int ret = clock_gettime(CLOCK_MONOTONIC, &ts);
    int ret = MPI_Wtime();
//    if (ret != 0) ERROR("clock_gettime");
//    return Interval((double)ts.tv_sec+(double)ts.tv_nsec/1e9, 0);
    return Interval(ret, 0);
    #endif
}

Interval Interval::cputime()
{
    #ifdef _OPENMP
    int tid = omp_get_thread_num();
    #else
    int tid = 0;
    #endif

    #ifdef __MACH__
    static double conv = -1.0;
    if (conv < 0)
    {
        mach_timebase_info_data_t timebase;
        mach_timebase_info(&timebase);
        conv = (double)timebase.numer / (double)timebase.denom;
    }
    uint64_t nsec = mach_absolute_time();
    return Interval(conv*(double)nsec/1e9, 0);
    #else
    timespec ts;
//    int ret = clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    int ret = MPI_Wtime();
//    if (ret != 0) ERROR("clock_gettime");
//    return Interval((double)ts.tv_sec+(double)ts.tv_nsec/1e9, 0);
    return Interval(ret, 0);
    #endif
}

bool Interval::operator<(const Interval& other) const
{
    return dt < other.dt;
}

Interval& Interval::operator+=(const Interval& other)
{
    dt += other.dt;
    flops += other.flops;
    return *this;
}

Interval& Interval::operator-=(const Interval& other)
{
    dt -= other.dt;
    flops -= other.flops;
    return *this;
}

Interval& Interval::operator*=(int m)
{
    dt *= m;
    return *this;
}

Interval& Interval::operator/=(int m)
{
    dt /= m;
    return *this;
}

Interval Interval::operator+(const Interval& other) const
{
    Interval r(*this);
    r += other;
    return r;
}

Interval Interval::operator-(const Interval& other) const
{
    Interval r(*this);
    r -= other;
    return r;
}

Interval Interval::operator*(int m) const
{
    Interval r(*this);
    r *= m;
    return r;
}

Interval Interval::operator/(int m) const
{
    Interval r(*this);
    r /= m;
    return r;
}

double Interval::seconds() const
{
    return dt;
}

double Interval::gflops() const
{
    return (double)flops/1e9/seconds();
}

std::list<Timer> Timer::timers;

bool Timer::operator<(const Timer& other) const
{
    return interval < other.interval;
}

Timer& Timer::timer(const std::string& name)
{
    for (list<Timer>::iterator it = timers.begin();it != timers.end();++it)
    {
        if (it->name == name) return *it;
    }

    timers.push_back(Timer(name));
    return timers.back();
}

void Timer::printTimers()
{
    int max_len = 0;
    for (list<Timer>::iterator it = timers.begin();it != timers.end();++it)
    {
        max_len = max(max_len, (int)it->name.size());
    }

    timers.sort();

    bool found = false;
    for (list<Timer>::iterator it = timers.begin();it != timers.end();++it)
    {
        double tot = it->seconds();
        long count = (long)it->count;
        double gflops = it->gflops();
        if (count > 0)
        {
            found = true;
            printf("%s:%*s %13.6f s %10ld x %11.6f gflops/sec\n", it->name.c_str(), (int)(max_len-it->name.size()), "", tot, count, gflops);
        }
    }
    if (found) printf("\n");
}

void Timer::clearTimers()
{
    for (list<Timer>::iterator it = timers.begin();it != timers.end();++it)
    {
        it->interval.dt = 0;
        it->interval.flops = 0;
        it->count = 0;
    }
}

void tic()
{
    #ifdef _OPENMP
    int tid = omp_get_thread_num();
    int ntd = (omp_in_parallel() ? 1 : omp_get_max_threads());
    #else
    int tid = 0;
    int ntd = 1;
    #endif
    for (int td = tid;td < tid+ntd;td++)
    {
        if (tics[td] == 0) tics[td] = new vector<Interval>();
        tics[td]->push_back(td == tid ? Interval::time() : Interval());
    }
}

Interval toc()
{
    #ifdef _OPENMP
    int tid = omp_get_thread_num();
    int ntd = (omp_in_parallel() ? 1 : omp_get_max_threads());
    #else
    int tid = 0;
    int ntd = 1;
    #endif
//    ASSERT(!tics[tid]->empty(), "Too many tocs");
    Interval dt = Interval::time();
    dt -= tics[tid]->back();
    tics[tid]->pop_back();
    for (int td = tid+1;td < tid+ntd;td++)
    {
//        ASSERT(!tics[td]->empty(), "Too many tocs");
        dt -= tics[td]->back();
        tics[td]->pop_back();
    }
    if (!tics[tid]->empty()) tics[tid]->back().flops -= dt.flops;
    return dt;
}

void cputic()
{
    #ifdef _OPENMP
    int tid = omp_get_thread_num();
    int ntd = (omp_in_parallel() ? 1 : omp_get_max_threads());
    #else
    int tid = 0;
    int ntd = 1;
    #endif
    for (int td = tid;td < tid+ntd;td++)
    {
        if (tics[td] == 0) tics[td] = new vector<Interval>();
        tics[td]->push_back(td == tid ? Interval::cputime() : Interval());
    }
}

Interval cputoc()
{
    #ifdef _OPENMP
    int tid = omp_get_thread_num();
    int ntd = (omp_in_parallel() ? 1 : omp_get_max_threads());
    #else
    int tid = 0;
    int ntd = 1;
    #endif
//    ASSERT(!tics[tid]->empty(), "Too many tocs");
    Interval dt = Interval::cputime();
    dt -= tics[tid]->back();
    tics[tid]->pop_back();
    for (int td = tid+1;td < tid+ntd;td++)
    {
//        ASSERT(!tics[td]->empty(), "Too many tocs");
        dt -= tics[td]->back();
        tics[td]->pop_back();
    }
    if (!tics[tid]->empty()) tics[tid]->back().flops -= dt.flops;
    return dt;
}

void do_flops(int64_t flops)
{
    #ifdef _OPENMP
    int tid = omp_get_thread_num();
    #else
    int tid = 0;
    #endif
    if (tics[tid] != 0 && !tics[tid]->empty())
        tics[tid]->back().flops -= flops;
}
