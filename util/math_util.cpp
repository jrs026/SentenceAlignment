#include "util/math_util.h"

#include "boost/math/special_functions/fpclassify.hpp"
#include "boost/math/special_functions/sign.hpp"

// Initialize the log table and its constants.
double MathUtil::log_add_inc = -LOG_ADD_MIN / LOG_ADD_TABLE_SIZE;
double MathUtil::inv_log_add_inc = LOG_ADD_TABLE_SIZE / -LOG_ADD_MIN;
double MathUtil::log_add_table[LOG_ADD_TABLE_SIZE+1];
bool MathUtil::use_approx = false;

bool MathUtil::ApproxEqual(double a, double b, int64_t max_ulps) {
  // Infinity check
  if (boost::math::isinf(a) || boost::math::isinf(b)) {
    return a == b;
  }
  // NaN check
  if ((boost::math::isnan(a)) || boost::math::isnan(b)) {
    return false;
  }

  int64_t a_int = *(int64_t*)&a;
  // Make aInt lexicographically ordered as a twos-complement int
  if (a_int < 0) {
    a_int = 0x80000000 - a_int;
  }
  // Make bInt lexicographically ordered as a twos-complement int
  int64_t b_int = *(int64_t*)&b;
  if (b_int < 0) {
    b_int = 0x80000000 - b_int;
  }

  // Now we can compare aInt and bInt to find out how far apart A and B
  // are.
  int64_t int_diff = abs(a_int - b_int);
  if (int_diff <= max_ulps) {
    return true;
  } 
  return false;
}

bool MathUtil::ApproxEqual(float a, float b, int32_t max_ulps) {
  // Infinity check
  if (boost::math::isinf(a) || boost::math::isinf(b)) {
    return a == b;
  }
  // NaN check
  if ((boost::math::isnan(a)) || boost::math::isnan(b)) {
    return false;
  }

  int32_t a_int = *(int32_t*)&a;
  // Make aInt lexicographically ordered as a twos-complement int
  if (a_int < 0) {
    a_int = 0x80000000 - a_int;
  }
  // Make bInt lexicographically ordered as a twos-complement int
  int32_t b_int = *(int32_t*)&b;
  if (b_int < 0) {
    b_int = 0x80000000 - b_int;
  }

  // Now we can compare aInt and bInt to find out how far apart A and B
  // are.
  int32_t int_diff = abs(a_int - b_int);
  if (int_diff <= max_ulps) {
    return true;
  } 
  return false;
}

void MathUtil::InitLogTable() {
  for(int i = 0; i <= LOG_ADD_TABLE_SIZE; i++) {
    log_add_table[i] = log1p(exp((i * log_add_inc) + LOG_ADD_MIN));
  }
  use_approx = true;
}

double math_util::Poisson(double lambda, int k) {
  double sum = 0;
  for (int i = 1; i <= k; ++i) {
    sum += log(i);
  }
  return exp((k * log(lambda)) - lambda - sum);
}

double math_util::Digamma(double x) {
  double result = 0, xx, xx2, xx4;
  assert(x > 0);
  for ( ; x < 7; ++x) {
    result -= 1 / x;
  }
  x -= 1.0 / 2.0;
  xx = 1.0 / x;
  xx2 = xx * xx;
  xx4 = xx2 * xx2;
  result += log(x) + (1./24.) * xx2 - (7.0/960.0) * xx4 + (31.0/8064.0) * xx4
      * xx2 - (127.0/30720.0) * xx4 * xx4;
  return result;
}
