#ifndef _TIME_HPP_
#define _TIME_HPP_

#include <list>
#include <mpi.h>
#include <string>
#include <time.h>
#include <vector>

#ifdef PROFILE

#define PROFILE_SECTION(name)                                                  \
  {                                                                            \
    static Timer &__timer = Timer::timer(name);                                \
    __timer.start();

#define PROFILE_FUNCTION PROFILE_SECTION(__func__)

#define PROFILE_STOP                                                           \
  __timer.stop();                                                              \
  }

#define PROFILE_RETURN                                                         \
  {                                                                            \
    __timer.stop();                                                            \
    return;                                                                    \
  }

#define PROFILE_FLOPS(n) do_flops(n)

#define PROFILE_MEMOPS(n) do_memops(n)

#else

#define PROFILE_FUNCTION

#define PROFILE_SECTION(name)

#define PROFILE_STOP

#define PROFILE_RETURN return;

#define PROFILE_FLOPS(n)

#define PROFILE_MEMOPS(n)

#endif

class Timer;

class Interval {
  friend class Timer;
  friend void do_flops(long long flops);
  friend void do_memops(long long memops);
  friend Interval toc();
  friend Interval cputoc();

protected:
  double dt;
  long long flops;
  long long memops;

  Interval(double start, long long flops, long long memops)
      : dt(start), flops(flops), memops(memops) {}

public:
  Interval() : dt(0), flops(0), memops(0) {}

  static Interval time();

  static Interval cputime();

  bool operator<(const Interval &other) const;

  Interval &operator+=(const Interval &other);

  Interval &operator-=(const Interval &other);

  Interval &operator*=(int m);

  Interval &operator/=(int m);

  Interval operator+(const Interval &other) const;

  Interval operator-(const Interval &other) const;

  Interval operator*(int m) const;

  Interval operator/(int m) const;

  double seconds() const;

  long long nflops() const;

  double gflops() const;

  double gmemops() const;
};

void tic();

Interval toc();

void cputic();

Interval cputoc();

void do_flops(long long flops);

void do_memops(long long memops);

class Timer {
protected:
  static std::list<Timer> timers;

  std::string name;
  Interval interval;
  long long count;

public:
  Timer(const std::string &name) : name(name), interval(), count(0) {}

  bool operator<(const Timer &other) const;

  static Timer &timer(const std::string &name);

  void start() { tic(); }

  void stop() {
    //            #ifdef _OPENMP
    //            if (!omp_in_parallel())
    //          {
    //            interval += toc()*omp_get_max_threads();
    //          count++;
    //    }
    //  else
    //            #pragma omp critical
    //            #endif
    //            {
    interval += toc();
    count++;
    //          }
  }

  static void printTimers();

  static void clearTimers();

  double seconds() const { return interval.seconds(); }

  double gflops() const { return interval.gflops(); }

  static long long nflops(const std::string &name);

  std::string getName() const { return name; }

  double gmemops() const { return interval.gmemops(); }
};

#endif
