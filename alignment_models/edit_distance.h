#ifndef _EDIT_DISTANCE_H_
#define _EDIT_DISTANCE_H_

// This class is used to test the monotonic alignment model. It learns
// stochastic edit distance.

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>
#include <tr1/unordered_map>
#include <utility>
#include <vector>

#include "alignment_models/monotonic_aligner.h"

using std::pair;
using std::string;
using std::vector;
using std::tr1::unordered_map;

// This is a joint model, so these parameters sum to 1. Epsilon is 0. eps:eps is
// used to store the stopping probability.
class EditDistanceParams {
 public:
  EditDistanceParams(int alphabet_size) : alphabet_size_(alphabet_size) {
    params_ = new double[(alphabet_size_+1)*(alphabet_size_)];
  }
  ~EditDistanceParams() {
    delete[] params_;
  }
  double& at(int i, int j) {
    assert((i < alphabet_size_) && (j < alphabet_size_));
    return (params_[(i*alphabet_size_) + j]);
  }
  const double& at(int i, int j) const {
    assert((i < alphabet_size_) && (j < alphabet_size_));
    return (params_[(i*alphabet_size_) + j]);
  }
  int alphabet_size() const { return alphabet_size_; }
 private:
  const int alphabet_size_; // Includes epsilon
  double* params_;
};

template <uint8_t order>
class StringPair : public SequencePair<order> {
 private:
  typedef typename SequencePair<order>::AlignmentState AlignmentState;
 public:
  StringPair() {}
  virtual ~StringPair() {}

  // Factory function. This function will clear the vectors passed to it, but
  // the parameter objects will remain intact.
  static StringPair<order>* CreateStringPair(
      EditDistanceParams* current_parameters,
      EditDistanceParams* expected_counts,
      vector<int>* input,
      vector<int>* output);

  // Inherited functions.
  virtual double GetScore(const AlignmentState& state,
                          const AlignmentOperation align_op) const;
  virtual double GetScoreAndUpdate(const AlignmentState& state,
                                   const AlignmentOperation align_op,
                                   const double expected_prob);
  virtual void ObservedArcUpdate(const AlignmentState& state,
                                 const AlignmentOperation align_op);

 private:
  // Return the indices in the parameter object corresponding to the given arc.
  void GetParameterIndices(const AlignmentState& state,
                           const AlignmentOperation align_op,
                           int* i, int* o) const;
  // The input and output strings stored as ints.
  vector<int> input_, output_;
  // The string pair does not own any of the following objects:
  // Alignment costs are taken from the current parameters.
  EditDistanceParams* current_parameters_;
  // The expected counts are updated during EM.
  EditDistanceParams* expected_counts_;
};

template <uint8_t order>
class EditDistanceModel {
 public:
  EditDistanceModel(int alphabet_size) : alphabet_size_(alphabet_size),
      params_(EditDistanceParams(alphabet_size_)) {
    // Uniform initialization
    double value = log(1.0 / (alphabet_size_ * alphabet_size_));
    for (int i = 0; i < alphabet_size_; ++i) {
      for (int j = 0; j < alphabet_size_; ++j) {
        params_.at(i, j) = value;
      }
    }
    // Epsilon must take the place of 0 in the vocabulary.
    vocab_.clear();
    vocab_['\0'] = 0;
  }
  ~EditDistanceModel() {}

  // Train on unlabeled data and return the perplexity on this data.
  double EM(const vector<pair<string, string> >& data, int iterations);

  // Align a string pair using the learned parameters.
  vector<AlignmentOperation>* Align(const pair<string, string>& pair) const;

  // Print the parameters of the model in a readable format
  void PrintParams(string* output) const;

 private:
  // Set all of the expected counts to 0 (in the log domain)
  void ClearCounts(EditDistanceParams* params) const;
  // Convert a string into a vector of vocabulary indices
  void ConvertToIndices(const string& str, vector<int>* indices);
  // Perform an M-step given the expected counts
  void UpdateParameters(const EditDistanceParams& expected_counts);
  const int alphabet_size_;
  EditDistanceParams params_;
  unordered_map<char, int> vocab_;
};

#endif
