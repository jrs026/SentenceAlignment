#ifndef __HMM_ALIGNER_H__
#define __HMM_ALIGNER_H__

#include <vector>

#include "alignment_models/packed_trie.h"
#include "util/parallel_corpus.h"
#include "util/vocab.h"

using std::vector;

// TODO: Create an abstract aligner class and have both Model1 and the HMM
// inherit from it.

class HMMLattice;
class TransProbs;

// This class contains an implementation of HMM-based word alignment

class HMMAligner {
 public:
  typedef ParallelCorpus::DocumentPair DocumentPair;
  typedef ParallelCorpus::Sentence Sentence;

  // Initialize the aligner with the smoothing parameters for the word
  // generation (emmision) and transition probabilities, the null word
  // generation probability, and the size of the window for storing distortion
  // probabilities.
  HMMAligner(double emission_smoothing, double transition_smoothing,
             double null_word_prob = 0.2, int window_size = 5)
             : emission_smoothing_(emission_smoothing),
               transition_smoothing_(transition_smoothing),
               null_word_prob_(log(null_word_prob)),
               window_size_(window_size) { }
  ~HMMAligner() {
    delete t_table_;
    delete expected_counts_;
    delete[] distortion_params_;
    delete[] distortion_counts_;
  }

  // Initialize the data structures for the given corpus. (for now, just the
  // global parameter vector). The T-Table is set to have uniform probabilities.
  void InitDataStructures(const vector<const ParallelCorpus*>& pc,
                          const Vocab& total_source_vocab,
                          const Vocab& total_target_vocab);

  // Returns the probability of the target sentence given the source.
  double ScorePair(const Sentence& source, const Sentence& target) const;
  // Returns the probability of the sentence pair and the best alignment.
  double ViterbiScorePair(const Sentence& source, const Sentence& target) const;

  // Clear (or initialize) the expected counts from the last E-Step.
  void ClearExpectedCounts();
  // Make an update to the expected counts from this sentence pair and return
  // the probability p(t|s).
  double EStep(const Sentence& source, const Sentence& target);
  // Update the parameters based on the expected counts. If variational is true,
  // the variational M-Step will be used.
  void MStep(bool variational);

  // Print the t-table to the given stream in a human readable format using the
  // given vocabularies.
  void PrintTTable(const Vocab& source_vocab, const Vocab& target_vocab,
      std::ostream& out) const;
  // Print the distortion parameters
  void PrintDistortionCosts(const Vocab& source_vocab,
      const Vocab& target_vocab, std::ostream& out) const;
  // Print both the t-table and the distortion probabilities
  void PrintModel(const Vocab& source_vocab, const Vocab& target_vocab,
      std::ostream& out) const;

  // Access and modify the expected counts
  PackedTrie* mutable_counts() { return expected_counts_; }
  // Access the probabilities
  const PackedTrie& t_table() const { return *t_table_; }

  // Accessors for some member variables
  double null_word_prob() const { return null_word_prob_; }
  int window_size() const { return window_size_; }

  // Accessor functions for the distortion probabilities and expected counts.
  // When attempting to access values outside of the window, these functions
  // will return the values at the edges.
  inline double dist_probs(int dist) const {
    return distortion_params_[dist_index(dist)];
  }
 private:
  inline double& dist_probs(int dist) {
    return distortion_params_[dist_index(dist)];
  }
  inline double& dist_counts(int dist) {
    return distortion_counts_[dist_index(dist)];
  }

  inline int dist_index(int dist) const {
    if (dist < -window_size_) {
      dist = -window_size_;
    } else if (dist > window_size_) {
      dist = window_size_;
    }
    return dist + window_size_;
  }
  const double emission_smoothing_;
  const double transition_smoothing_;
  const double null_word_prob_;
  const int window_size_;
  // Probabilities and expected counts for the distortion model.
  double* distortion_params_;
  double* distortion_counts_;

  int source_vocab_size_;
  int target_vocab_size_;
  // The T-Table, holds log p(t|s).
  PackedTrie* t_table_;
  // Holds the expected counts for the E-Step.
  PackedTrie* expected_counts_;
};

// An internal lattice class for the HMM
class HMMLattice {
 public:
  HMMLattice(int source_length, int target_length);
  ~HMMLattice();
  
  // Lattice accessor.
  // Source pos can be from -1 to source length
  // Target pos can be from 0 to target length
  inline double& at(int source_pos, int target_pos, bool null) {
    int s = source_pos + 1;
    if (null) {
      s += source_length_ + 1;
    }
    return lattice_[target_pos][s];
  }
 private:
  int source_length_;
  int target_length_;
  double** lattice_;
};

// A class for storing and accessing the transition probabilities for a given
// sentence.
class TransProbs {
 public:
  TransProbs(int source_length, const HMMAligner& hmm);
  ~TransProbs();

  inline double at(int prev, int next) const {
    return trans_probs_[prev + 1][next];
  }
 private:
  int source_length_;
  double** trans_probs_;
};

#endif
