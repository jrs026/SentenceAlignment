#include "util/vocab.h"

#include <utility>

using std::string;
using std::tr1::unordered_map;
using std::vector;

int Vocab::AddWord(const string& word) {
  index_cit it = index_.find(word);
  if (it != index_.end()) {
    return it->second;
  } else {
    const int index = rev_index_.size();
    index_.insert(std::make_pair(word, index));
    rev_index_.push_back(word);
    return index;
  }
}

int Vocab::GetIndex(const string& word) const {
  index_cit it = index_.find(word);
  if (it != index_.end()) {
    return it->second;
  } else {
    return 0;
  }
}

string Vocab::GetWord(const int index) const {
  return rev_index_.at(index);
}

void Vocab::Merge(const Vocab& vocab) {
  for (int i = 0; i < vocab.rev_index_.size(); ++i) {
    AddWord(vocab.rev_index_.at(i));
  }
}
