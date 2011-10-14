#ifndef _VOCAB_H_
#define _VOCAB_H_

// This class defines a basic vocabulary which maps strings to integers.

#include <string>
#include <tr1/unordered_map>
#include <vector>

using std::string;
using std::tr1::unordered_map;
using std::vector;

class Vocab {
 private:
  typedef unordered_map<string, int>::const_iterator index_cit;

 public:
  Vocab() {
    // Add OOV to the index
    index_["OOV"] = 0;
    rev_index_.push_back("OOV");
  }
  ~Vocab() {}

  // Returns the index of a word, adding it to the vocabulary if it doesn't
  // exist.
  int AddWord(const string& word);
  // Returns the index of a word, or 0 if the word does not exist.
  int GetIndex(const string& word) const;
  // Returns the word corresponding to the index.
  string GetWord(const int index) const;
  // Adds all entries in the given vocabulary to this one.
  void Merge(const Vocab& vocab);
  int size() const { return index_.size(); }

 private:
  // Maps words to integers
  unordered_map<string, int> index_;
  // Maps integers to words
  vector<string> rev_index_;
};

#endif
