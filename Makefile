ifdef DEBUG
OPT=-g -std=c++0x
else
OPT=-O3 -ffast-math -funroll-loops #-std=c++0x -mtune=native -march=native
endif

ifdef PROF
OPTS=-pg -fno-inline $(OPT)
else
OPTS=$(OPT)
endif

INCS= -I. -I$(BOOST_ROOT)/include
LIBS =  -L$(BOOST_ROOT)/lib -L/usr/lib
LDFLAGS = -lm -lpthread -ldl -lboost_program_options -lboost_filesystem -lboost_serialization -lpthread -lboost_thread -lboost_system

CXX = /usr/bin/g++

CXXFLAGS=$(OPTS) $(INCS)

SRC_FILES = \
	alignment_models/monotonic_aligner.cpp \
	util/math_util.cpp \
	main.cpp
	
OBJ_FILES = $(SRC_FILES:%.cpp=%.o)
TARGET = monotonic_aligner_test

%.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) -o $@ -c $<

all: $(OBJ_FILES)
	$(CXX) $(CXXFLAGS) $(LIBS) $(OBJ_FILES) -o $(TARGET) $(LDFLAGS)

clean:
	rm $(OBJ_FILES) $(TARGET)
