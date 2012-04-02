#ifndef _MODEL_1_H_
#define _MODEL_1_H_

// This class contains an implementation of IBM Model 1 for word alignment. 

#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <tr1/unordered_map>

#include "alignment_models/packed_trie.h"
#include "util/parallel_corpus.h"
#include "util/vocab.h"

using std::string;
using std::vector;

class Model1 {
 public:
  typedef ParallelCorpus::DocumentPair DocumentPair;
  typedef ParallelCorpus::Sentence Sentence;

  // Alpha is the symmetric dirichlet prior on the t-table probabilities.
  Model1(double alpha = 1.0) : alpha_(alpha) {}
  ~Model1() {
    delete t_table_;
    delete expected_counts_;
  }

  // Initialize the data structures for the given corpus. (for now, just the
  // global parameter vector). The T-Table is set to have uniform probabilities.
  void InitDataStructures(const vector<const ParallelCorpus*>& pc,
                          const Vocab& total_source_vocab,
                          const Vocab& total_target_vocab);
  // Initialize the data structures from a printed t-table and return the
  // vocabularies.
  void InitFromFile(const string& filename, Vocab* source_vocab,
      Vocab* target_vocab);
  // Initialize the data structures from a binary t-table and vocabulary files.
  void InitFromBinaryFile(const string& t_table_file,
                          const string& source_vocab_file,
                          const string& target_vocab_file,
                          Vocab* source_vocab,
                          Vocab* target_vocab);

  
  // Returns the probability of the target sentence given the source.
  double ScorePair(const Sentence& source, const Sentence& target) const;
  // Returns the probability of the sentence pair and the best alignment.
  double ViterbiScorePair(const Sentence& source, const Sentence& target) const;

  // Compute the percentage of target words "covered" by some source word
  // (log p(t|s) must be greater than log_word_cutoff for some s).
  // If covered_unk is true, unknown words will be considered covered.
  // If ignored_unk is true, unknown words will not be considered in the
  // percentage of covered words (overridden by covered_unk).
  double ComputeCoverage(const Sentence& source, const Sentence& target,
      double log_word_cutoff, bool covered_unk, bool ignored_unk) const;

  // Clear (or initialize) the expected counts from the last E-Step.
  void ClearExpectedCounts();
  // Make an update to the expected counts from this sentence pair and return
  // the probability p(t|s). The weight of the example is in the log domain.
  double EStep(const Sentence& source, const Sentence& target, double weight);
  // Update the parameters based on the expected counts. If variational is true,
  // the variational M-Step will be used.
  void MStep(bool variational);

  // Print the t-table to the given stream in a human readable format using the
  // given vocabularies.
  void PrintTTable(const Vocab& source_vocab, const Vocab& target_vocab,
      std::ostream& out) const;

  // Write the TTable in a binary format and the vocabularies to a
  // human readable format.
  void WriteBinary(const string& t_table_file,
                   const string& source_vocab_file,
                   const string& target_vocab_file,
                   const Vocab& source_vocab,
                   const Vocab& target_vocab) const;

  // Access and modify the expected counts
  PackedTrie* mutable_counts() { return expected_counts_; }
  // Access the probabilities
  const PackedTrie& t_table() const { return *t_table_; }
 
 private:
  const double alpha_;
  // Vocabulary objects for the source and target sides
  const Vocab* source_vocab_;
  const Vocab* target_vocab_;
  // The T-Table, holds log p(t|s).
  PackedTrie* t_table_;
  // Holds the expected counts for the E-Step.
  PackedTrie* expected_counts_;
};

#endif
