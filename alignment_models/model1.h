#ifndef _MODEL_1_H_
#define _MODEL_1_H_

// This class contains an implementation of IBM Model 1 for word alignment. 

#include <vector>
#include <tr1/unordered_map>
#include <utility>

#include "util/parallel_corpus.h"
#include "util/vocab.h"

typedef std::pair<int, int> WordPair;

// Hash function for the T-Table.
// TODO: If std::size_t is 8 bytes, this works well. If not, 
struct WordPairHash {
  std::size_t operator()(const WordPair& wp) const {
    return (static_cast<std::size_t>(wp.first) << sizeof(std::size_t) * 4)
        + wp.second;
  }
};

class Model1 {
 public:
  typedef ParallelCorpus::DocumentPair DocumentPair;
  typedef ParallelCorpus::Sentence Sentence;
  // TODO: Describe the storage for expected counts.
  typedef std::tr1::unordered_map<WordPair, double, WordPairHash> TTable;

  Model1() {}
  ~Model1() {}

  // Initialize the data structures for the given corpus. (for now, just the
  // global parameter vector). The T-Table is set to have uniform probabilities.
  void InitDataStructures(const ParallelCorpus& pc);
  
  // Returns the probability of the target sentence pair given the source.
  double ScorePair(const Sentence& source, const Sentence& target) const;

  // Clear (or initialize) the expected counts from the last E-Step.
  void ClearExpectedCounts();
  // Make an update to the expected counts from this sentence pair and return
  // the probability p(t|s). The weight of the example is in the log domain.
  double EStep(const Sentence& source, const Sentence& target, double weight);
  // Update the parameters based on the expected counts.
  void MStep();
 
 private:
  // Used for accessing entries in a TTable. It is assumed that all of the
  // entries are already present.
  inline double& lookup(TTable* table, int source_word, int target_word) {
    return (*table)[std::make_pair(source_word, target_word)];
  }
  inline const double& lookup(
      const TTable* table, int source_word, int target_word) const {
    return table->find(std::make_pair(source_word, target_word))->second;
  }
  // Sizes of the source and target vocabularies, set during InitDataStructures.
  int source_vocab_size_;
  int target_vocab_size_;
  // The T-Table, holds log p(t|s).
  TTable t_table_;
  // Holds the expected counts for the E-Step.
  TTable expected_counts_;
};

#endif
