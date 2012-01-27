#ifndef _UTIL_H_
#define _UTIL_H_

// Some basic utility functions

#include <sstream>
#include <string>

namespace util {

template<typename T>
std::string ToString(const T& t) {
  std::stringstream sstr;
  sstr << t;
  return sstr.str();
}

}  // end namespace util

#endif
