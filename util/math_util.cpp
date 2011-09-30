#include "util/math_util.h"

#include "boost/math/special_functions/fpclassify.hpp"
#include "boost/math/special_functions/sign.hpp"

// Initialize the log table and its constants.
double MathUtil::log_add_inc = -LOG_ADD_MIN / LOG_ADD_TABLE_SIZE;
double MathUtil::inv_log_add_inc = LOG_ADD_TABLE_SIZE / -LOG_ADD_MIN;
double MathUtil::log_add_table[LOG_ADD_TABLE_SIZE+1];

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
  for(int i = 0; i <= LOG_ADD_TABLE_SIZE; i++)
    log_add_table[i] = log1p(exp((i * log_add_inc) + LOG_ADD_MIN));
}
