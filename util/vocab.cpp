#include "util/vocab.h"

#include <fstream>
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

string Vocab::ToText(const vector<int>& sentence) const {
  string result = "";
  if (sentence.size() > 0) {
    result = GetWord(sentence.at(0));
  }
  for (int i = 1; i < sentence.size(); ++i) {
    result += " " + GetWord(sentence.at(i));
  }
  return result;
}

bool Vocab::Read(const string& filename) {
  std::ifstream in(filename.c_str());
  if (!in.good()) {
    return false;
  }

  unordered_map<string, int> new_index;
  vector<string> new_rev_index;
  string line;
  while (getline(in, line)) {
    index_cit cit = new_index.find(line);
    if (cit != new_index.end()) {
      return false; // duplicate entry
    }
    new_index[line] = new_rev_index.size();
    new_rev_index.push_back(line);
  }
  in.close();
  index_.swap(new_index);
  rev_index_.swap(new_rev_index);
  return true;
}

void Vocab::Write(const string& filename) const {
  std::ofstream out(filename.c_str());
  for (int i = 0; i < rev_index_.size(); ++i) {
    out << rev_index_.at(i) << std::endl;
  }
  out.close();
}
