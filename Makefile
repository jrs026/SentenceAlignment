ifdef DEBUG
OPT=-g -std=c++0x
else
OPT=-O3 -ffast-math -funroll-loops -DNDEBUG -std=c++0x #-mtune=native -march=native
endif

ifdef PROF
OPTS=-pg $(OPT) #-fno-inline $(OPT)
else
OPTS=$(OPT)
endif

INCS = -I. -I$(BOOST_ROOT)
LIBS =  -L$(BOOST_LIB)
LDFLAGS = -lm -lpthread -ldl -lboost_program_options -lboost_filesystem -lboost_serialization -lpthread -lboost_thread -lboost_system

CXX = /usr/bin/g++

CXXFLAGS=$(OPTS) $(INCS)

SRC_FILES = \
  alignment_models/document_aligner.cpp \
  alignment_models/edit_distance.cpp \
  alignment_models/hmm_aligner.cpp \
  alignment_models/model1.cpp \
	alignment_models/monotonic_aligner.cpp \
  alignment_models/packed_trie.cpp \
	util/math_util.cpp \
  util/nullbuf.cpp \
  util/parallel_corpus.cpp \
  util/vocab.cpp

TEST_FILE = test.cpp
MAIN_FILE = main.cpp
	
OBJ_FILES = $(SRC_FILES:%.cpp=%.o)
TEST_OBJ = $(TEST_FILE:%.cpp=%.o)
MAIN_OBJ = $(MAIN_FILE:%.cpp=%.o)
TEST_TARGET = monotonic_aligner_test
MAIN_TARGET = doc_aligner

%.o: %.cpp %.h
	$(CXX) $(CXXFLAGS) -o $@ -c $<

all: $(OBJ_FILES) $(TEST_OBJ) $(MAIN_OBJ)
	$(CXX) $(CXXFLAGS) $(LIBS) $(OBJ_FILES) $(TEST_OBJ) -o $(TEST_TARGET) $(LDFLAGS)
	$(CXX) $(CXXFLAGS) $(LIBS) $(OBJ_FILES) $(MAIN_OBJ) -o $(MAIN_TARGET) $(LDFLAGS)

clean:
	rm $(OBJ_FILES) $(TEST_OBJ) $(MAIN_OBJ) $(TEST_TARGET) $(MAIN_TARGET)
