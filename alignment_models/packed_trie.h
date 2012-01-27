#ifndef __PACKED_TRIE_H__
#define __PACKED_TRIE_H__

#include <cassert>
#include <limits>
#include <set>
#include <vector>
#include <tr1/unordered_set>

#include "util/parallel_corpus.h"

using std::set;
using std::string;
using std::vector;

using std::tr1::unordered_set;

class Vocab;

// TODO: Make sure that the data structures are not initialized more than once.

// Packed trie: a data structure for storing the parameters of a word alignment
// model. This is not a generic packed trie.
// The trie must be initialized with all entries when created.

// Once the range is sufficiently small, linear search is used.
#define LINEAR_SEARCH_LIMIT 8

class PackedTrie {
 public:
  typedef ParallelCorpus::DocumentPair DocumentPair;
  typedef ParallelCorpus::Sentence Sentence;

  PackedTrie() {}
  // Copy constructor. NOTE: This does not copy all fields, so the trie this is
  // copying from must stay alive for the life of this object.
  PackedTrie(const PackedTrie& trie);
  ~PackedTrie() {
    if (!is_dependent_) {
      delete[] offsets_;
      delete[] target_words_;
    }
    delete[] data_;
  }

  // Initialize the data structures of the packed trie from the given parallel
  // corpora. The trie will store t-table entries for any source/target word
  // that appears in a document pair.
  // It is assumed that the null word (index 0) can generate all target words no
  // matter what targets_per_source contains.
  void InitializeFromCorpus(const vector<const ParallelCorpus*>& pcs,
                            const Vocab& total_source_vocab,
                            const Vocab& total_target_vocab);
  
  // Binary search function used by accessors
  inline int FindIndex(int source_word, int target_word) const {
    RangeCheck(source_word, target_word);
    if (source_word == 0) {
      return target_word;
    }
    // OOV entry lookups
    if ((target_word == 0) || (target_word >= target_count_)) {
      return offsets_[source_word];
    }
    int lower = offsets_[source_word];
    int upper = offsets_[source_word + 1] - 1;
    while (lower <= upper) {
      if (upper - lower <= LINEAR_SEARCH_LIMIT) {
        while (lower <= upper) {
          if (target_word == target_words_[lower]) {
            return lower;
          }
          lower++;
        }
      } else {
        int mid = (lower + upper) >> 1;
        if (target_word == target_words_[mid]) {
          return mid;
        } else if (target_word < target_words_[mid]) {
          upper = mid - 1;
        } else {
          lower = mid + 1;
        }
      }
    }
    // Not found, must be OOV (for this source word)
    return offsets_[source_word];
  }
  // Accessor functions to the entries in the trie.
  inline double& Lookup(int source_word, int target_word) {
    return data_[FindIndex(source_word, target_word)];
  }
  inline double Prob(int source_word, int target_word) const {
    // Use a uniform probability when the source word is OOV:
    if (source_word >= source_count_) {
      return log(1.0 / target_count_);
    } else {
      return data_[FindIndex(source_word, target_word)];
    }
  }

  // Direct accessors for more efficient lookups while iterating
  inline int Offset(int s) const {
    return offsets_[s];
  }
  inline int TargetWord(int index) const {
    return target_words_[index];
  }
  inline double Data(int index) const {
    return data_[index];
  }
  inline double& Data(int index) {
    return data_[index];
  }

  // Re-initialize all entries to the given value.
  void Clear(double initial_value = -std::numeric_limits<double>::max());

  // Print the t-table to disk in a human readable format.
  void Print(const Vocab& source_vocab, const Vocab& target_vocab,
      std::ostream& out) const;
  // Read the human readable format. This will initialize the data structures
  // and the vocabularies (if it returns true).
  bool Read(const string& filename, Vocab* source_vocab, Vocab* target_vocab);

  // Read and write the packed trie to disk in binary format.
  bool ReadBinary(const string& filename);
  void WriteBinary(const string& filename) const;

  // Used for sorting sections of the target_words_ array
  static int int_cmp(const void *x, const void *y)
  {
    double xx = *(int*)x, yy = *(int*)y;
    if (xx < yy) return -1;
    if (xx > yy) return  1;
    return 0;
  }
 private:
  // Used to detect out-of-bounds array access
  inline void RangeCheck(int source_word, int target_word) const {
    assert((source_word >= 0) && (source_word < source_count_));
    assert(target_word >= 0);
  }
  // The total number of source and target words.
  int source_count_;
  int target_count_;
  // The entries in the trie are stored in a large flat array (data_). To find
  // an entry for a source/target word pair, the offset_ array gives the
  // location of the target entries given the source word. Then, a binary search
  // can be done between offsets[s] and (offsets[s+1] - 1).
  int* offsets_; // source_words_ + 1 entries
  int total_size_; // Total number of entries in the trie
  int* target_words_; // total_size_ entries
  double* data_; // total_size_ entries

  // This variable is true if the offsets_ and target_words_ arrays come from
  // another class.
  bool is_dependent_;
};

#endif
