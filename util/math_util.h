#ifndef _MATH_UTIL_H_
#define _MATH_UTIL_H_

// MathUtil.h
//
// Some utility functions for working with probabilities, log-domain values, and
// other numeric issues.

#include <cmath>
#include <stdint.h>

class MathUtil {
 private:
  // Log table constants
#define LOG_ADD_TABLE_SIZE 60000 // Number of entries in the table
#define LOG_ADD_MIN -64 // Smallest value for b-a
  static double log_add_inc;
  static double inv_log_add_inc;
  static double log_add_table[LOG_ADD_TABLE_SIZE+1];
 
 public:
  // Approximate equality functions for floating point numbers taken from
  // http://www.cygnus-software.com/papers/comparingfloats/comparingfloats.htm
#define MAX_DOUBLE_ULPS 16
  static bool ApproxEqual(
      double a, double b, int64_t maxUlps = MAX_DOUBLE_ULPS);
  static bool ApproxEqual(
      float a, float b, int32_t maxUlps = MAX_DOUBLE_ULPS / 2);

  // An exact implementation for adding two numbers in the log domain
  static inline double LogAdd(double a, double b) {
    if (b > a) {
      double temp = a;
      a = b;
      b = temp;
    }
    return a + log1p(exp(b - a));
  }

  // Populates the log add table. Must be run before ApproxLogAdd can be used.
  static void InitLogTable();

  // Add the two numbers in the log domain using the log add table.
  static inline double ApproxLogAdd(double a, double b) {
    if (b > a) {
      double temp = a;
      a = b;
      b = temp;
    }
    double neg_diff = (b - a) - LOG_ADD_MIN;
    if (neg_diff < 0.0) {
      return a;
    }
    return a + log_add_table[(int)(neg_diff * inv_log_add_inc)];
  }
};

#endif
