#ifndef __NULLBUF_H__
#define __NULLBUF_H__

// Null buffers and output streams, taken from:
// http://groups.google.com/group/comp.lang.c++.moderated/msg/6125e4267ce31d7a?pli=1
// and
// http://stackoverflow.com/questions/6240950/platform-independent-dev-null-in-c

#include <iostream>

template<typename Ch, typename Traits = std::char_traits<Ch> >
struct basic_nullbuf : std::basic_streambuf<Ch, Traits> {
  typedef std::basic_streambuf<Ch, Traits> base_type;
  typedef typename base_type::int_type int_type;
  typedef typename base_type::traits_type traits_type;

  virtual int_type overflow(int_type c) {
    return traits_type::not_eof(c);
  }
};

// convenient typedefs
typedef basic_nullbuf<char> nullbuf;
typedef basic_nullbuf<wchar_t> wnullbuf;

extern std::ostream cnull;
extern std::wostream wcnull;

#endif
