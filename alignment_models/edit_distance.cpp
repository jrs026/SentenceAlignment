#include "alignment_models/edit_distance.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <sstream>
#include <vector>

#include "util/math_util.h"

using std::pair;
using std::string;
using std::vector;

// ---------------------------------------------------------------------------
// StringPair

template <uint8_t order>
StringPair<order>* StringPair<order>::CreateStringPair(
    EditDistanceParams* current_parameters,
    EditDistanceParams* expected_counts,
    vector<int>* input,
    vector<int>* output) {
  StringPair<order>* result = new StringPair<order>();
  result->current_parameters_ = current_parameters;
  result->expected_counts_ = expected_counts;
  result->input_.swap(*input);
  result->output_.swap(*output);
  result->input_length_ = result->input_.size();
  result->output_length_ = result->output_.size();
  return result;
}

template<uint8_t order>
double StringPair<order>::GetScore(const AlignmentState& state,
                                   const AlignmentOperation align_op) const {
  int i, j;
  GetParameterIndices(state, align_op, &i, &j);
  return current_parameters_->at(i, j);
}

template<uint8_t order>
double StringPair<order>::GetScoreAndUpdate(const AlignmentState& state,
                                            const AlignmentOperation align_op,
                                            const double expected_prob) {
  int i, j;
  GetParameterIndices(state, align_op, &i, &j);
  double score = current_parameters_->at(i, j);
  // This includes the stopping probability, which the monotonic aligner doesn't
  // capture.
  expected_counts_->at(i, j) = MathUtil::LogAdd(expected_counts_->at(i, j),
      score + expected_prob + current_parameters_->at(0, 0));
  return score;
}

template<uint8_t order>
void StringPair<order>::ObservedArcUpdate(const AlignmentState& state,
                                          const AlignmentOperation align_op) {
  int i, j;
  GetParameterIndices(state, align_op, &i, &j);
  // Adding 1 to the expected counts (in the log domain)
  expected_counts_->at(i, j) = MathUtil::LogAdd(expected_counts_->at(i, j),
                                                0.0);
}

template<uint8_t order>
void StringPair<order>::GetParameterIndices(const AlignmentState& state,
                                            const AlignmentOperation align_op,
                                            int* i, int* o) const {
  int i1, i2, o1, o2;
  this->SpanFromAlignmentArc(state, align_op, &i1, &i2, &o1, &o2);
  if ((i2 - i1 == 1) && (o2 - o1 == 1)) {
    // Substitution
    *i = input_.at(i1);
    *o = output_.at(o1);
  } else if ((i2 - i1 == 1) && (o2 - o1 == 0)) {
    // Deletion
    *i = input_.at(i1);
    *o = 0;
  } else if ((i2 - i1 == 0) && (o2 - o1 == 1)) {
    // Insertion
    *i = 0;
    *o = output_.at(o1);
  } else {
    std::cerr << "Illegal alignment for string pairs." << std::endl;
    exit(-1);
  }
}

// ---------------------------------------------------------------------------
// EditDistanceModel

// Train on unlabeled data and return the perplexity on this data.
template<uint8_t order>
double EditDistanceModel<order>::EM(const vector<pair<string, string> >& data,
                                    int iterations) {
  EditDistanceParams* expected_counts = new EditDistanceParams(alphabet_size_);
  // Convert the data into sequence pairs
  vector<SequencePair<order>* > string_pairs;
  for (int i = 0; i < data.size(); ++i) {
    vector<int> input, output;
    ConvertToIndices(data.at(i).first, &input);
    ConvertToIndices(data.at(i).second, &output);
    string_pairs.push_back(StringPair<order>::CreateStringPair(
        &params_, expected_counts, &input, &output));
  }
  MonotonicAligner<order> aligner(2);
  double perplexity = 0.0;
  for (int i = 0; i < iterations; ++i) {
    ClearCounts(expected_counts);
    // E step: run forward backward on all string pairs and collect the counts
    perplexity = 0.0;
    for (int j = 0; j < string_pairs.size(); ++j) {
      perplexity += aligner.ForwardBackward(string_pairs[j]);
      // Include the stopping cost
      perplexity += params_.at(0, 0);
    }
    // Set the expected count of the end-of-string
    // TODO: this needs to be double-checked
    expected_counts->at(0, 0) = params_.at(0, 0) + log(string_pairs.size());
    // M step: update the parameters from the expected counts
    UpdateParameters(*expected_counts);
    std::cout << "Iteration " << i << " perplexity: " << perplexity
        << std::endl;
  }

  delete expected_counts;
  for (int i = 0; i < string_pairs.size(); ++i) {
    delete string_pairs[i];
  }
  return perplexity;
}

// Align a string pair using the learned parameters.
template<uint8_t order>
vector<AlignmentOperation>* EditDistanceModel<order>::Align(
    const pair<string, string>& pair) const {
  // TODO
  vector<AlignmentOperation>* result = new vector<AlignmentOperation>();
  return result;
}

template<uint8_t order>
void EditDistanceModel<order>::PrintParams(string* output) const {
  output->clear();
  std::stringstream sstr;
  for (int i = 0; i < alphabet_size_; ++i) {
    for (int j = 0; j < alphabet_size_; ++j) {
      sstr << i << " " << j << " : " << params_.at(i, j) << std::endl; 
    }
  }
  *output = sstr.str();
}

template<uint8_t order>
void EditDistanceModel<order>::ClearCounts(EditDistanceParams* params) const {
  for (int i = 0; i < alphabet_size_; ++i) {
    for (int j = 0; j < alphabet_size_; ++j) {
      params->at(i, j) = -std::numeric_limits<double>::max();
    }
  }
}

template<uint8_t order>
void EditDistanceModel<order>::ConvertToIndices(
    const string& str, vector<int>* indices) {
  indices->clear();
  for (int i = 0; i < str.length(); ++i) {
    std::tr1::unordered_map<char, int>::const_iterator it =
        vocab_.find(str.at(i));
    if (it == vocab_.end()) {
      indices->push_back(vocab_.size());
      vocab_[str.at(i)] = indices->at(i);
    } else {
      indices->push_back(it->second);
    }
  }
  // Make sure the alphabet size wasn't exceeded.
  if (alphabet_size_ < vocab_.size()) {
    std::cerr << "Alphabet size exceeded" << std::endl;
    for (std::tr1::unordered_map<char, int>::iterator it = vocab_.begin();
         it != vocab_.end(); ++it) {
      std::cout << (int) it->first << " " << it->second << std::endl;
    }
    exit(-1);
  }
}

template<uint8_t order>
void EditDistanceModel<order>::UpdateParameters(
    const EditDistanceParams& expected_counts) {
  double norm = -std::numeric_limits<double>::max();
  for (int i = 0; i < alphabet_size_; ++i) {
    for (int j = 0; j < alphabet_size_; ++j) {
      norm = MathUtil::LogAdd(norm, expected_counts.at(i, j));
    }
  }
  for (int i = 0; i < alphabet_size_; ++i) {
    for (int j = 0; j < alphabet_size_; ++j) {
      params_.at(i, j) = expected_counts.at(i, j) - norm;
    }
  }
}

template class StringPair<0>;
template class EditDistanceModel<0>;
