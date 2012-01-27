#include "alignment_models/packed_trie.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <tr1/unordered_set>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>

#include "util/vocab.h"

using std::cout;
using std::endl;
using std::ios;
using std::string;
using std::tr1::unordered_set;

void PackedTrie::InitializeFromCorpus(const vector<const ParallelCorpus*>& pcs,
                                      const Vocab& total_source_vocab,
                                      const Vocab& total_target_vocab) {
  is_dependent_ = false;
  source_count_ = total_source_vocab.size();
  target_count_ = total_target_vocab.size();

  // For each source word, keep track of how many possible target words it can
  // generate. The null word can generate anything.
  vector<unordered_set<int> > targets_per_source;
  targets_per_source.resize(source_count_);
  // Each source word can generate the OOV word, which always has id 0.
  for (int s = 0; s < source_count_; ++s) {
    targets_per_source[s].insert(0);
  }
  for (int i = 0; i < pcs.size(); ++i) {
    for (int j = 0; j < pcs.at(i)->size(); ++j) {
      // Put all of the source and target words in the document pair into sets.
      const DocumentPair& doc_pair = pcs.at(i)->GetDocPair(j);
      
      unordered_set<int> source_words, target_words;
      for (int t = 0; t < doc_pair.second.size(); ++t) {
        const Sentence& sentence = doc_pair.second[t];
        for (int w = 0; w < sentence.size(); ++w) {
          target_words.insert(sentence[w]);
        }
      }
      for (int s = 0; s < doc_pair.first.size(); ++s) {
        const Sentence& sentence = doc_pair.first[s];
        for (int w = 0; w < sentence.size(); ++w) {
          source_words.insert(sentence[w]);
        }
      }
      unordered_set<int>::const_iterator s_it, t_it;
      for (s_it = source_words.begin(); s_it != source_words.end(); ++s_it) {
        for (t_it = target_words.begin(); t_it != target_words.end(); ++t_it) {
          targets_per_source[*s_it].insert(*t_it);
        }
      }
    }
  }
  assert(targets_per_source.size() > 0);

  offsets_ = new int[source_count_ + 1];

  // Null word
  offsets_[0] = 0;
  total_size_ = target_count_;
  for (int i = 1; i < targets_per_source.size(); ++i) {
    offsets_[i] = total_size_;
    total_size_ += targets_per_source[i].size();
  }
  offsets_[source_count_] = total_size_;

  target_words_ = new int[total_size_];
  data_ = new double[total_size_];

  // Add the entries
  // Null word entries
  for (int i = 0; i < target_count_; ++i) {
    target_words_[i] = i;
    data_[i] = log(1.0 / target_count_);
  }
  for (int s = 1; s < targets_per_source.size(); ++s) {
    int index = offsets_[s];
    unordered_set<int>::const_iterator it = targets_per_source[s].begin();
    for ( ; it != targets_per_source[s].end(); ++it) {
      target_words_[index] = *it;
      data_[index] = log(1.0 / targets_per_source[s].size());
      ++index;
    }
    qsort(target_words_ + offsets_[s],
          offsets_[s+1] - offsets_[s],
          sizeof(int),
          PackedTrie::int_cmp);
  }
  for (int s = 0; s < source_count_; ++s) {
    int last = -1;
    //cout << "Source word: " << total_source_vocab.GetWord(s) << endl;
    for (int i = offsets_[s]; i < offsets_[s+1]; ++i) {
      //cout << total_target_vocab.GetWord(target_words_[i]) << ":"
      //     << target_words_[i] << ":" << data_[i] << " ";
      if (last >= target_words_[i]) {
        cout << "Error on source word " << s << endl;
      }
      last = target_words_[i];
    }
    //cout << endl;
  }
}

PackedTrie::PackedTrie(const PackedTrie& trie) {
  is_dependent_ = true;
  source_count_ = trie.source_count_;
  target_count_ = trie.target_count_;
  total_size_ = trie.total_size_;
  data_ = new double[total_size_];
  for (int i = 0; i < total_size_; ++i) {
    data_[i] = trie.data_[i];
  }
  // This is why trie must remain alive when using the copy constructor.
  offsets_ = trie.offsets_;
  target_words_ = trie.target_words_;
}

void PackedTrie::Clear(double initial_value) {
  for (int i = 0; i < total_size_; ++i) {
    data_[i] = initial_value;
  }
}

void PackedTrie::Print(const Vocab& source_vocab, const Vocab& target_vocab,
    std::ostream& out) const {
  for (int s = 0; s < source_count_; ++s) {
    for (int i = offsets_[s]; i < offsets_[s + 1]; ++i) {
      out << source_vocab.GetWord(s) << "\t"
          << target_vocab.GetWord(target_words_[i]) << "\t"
          << exp(data_[i]) << std::endl;
    }
  }
}

bool PackedTrie::Read(const string& filename, Vocab* source_vocab,
    Vocab* target_vocab) {
  std::ifstream in(filename.c_str());
  if (!in.good()) {
    return false;
  }
  typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
  boost::char_separator<char> sep("\t");
  // First pass: find the total size and the source and target counts
  string line;
  total_size_ = 0;
  while (getline(in, line)) {
    tokenizer line_tokenizer(line, sep);
    int i = 0;
    for (tokenizer::iterator it = line_tokenizer.begin();
         it != line_tokenizer.end(); ++it) {
      if (i == 0) { // Source word
        source_vocab->AddWord(*it);
      } else if (i == 1) { // Target word
        target_vocab->AddWord(*it);
      } else if (i == 2) { // Probability
        // Nothing needed here in the first pass.
      }
      ++i;
    }
    assert(i == 3); // Each line should have exactly 3 tokens.
    ++total_size_; 
  }
  // Initialize the main data structures
  source_count_ = source_vocab->size();
  target_count_ = target_vocab->size();
  offsets_ = new int[source_count_ + 1];
  target_words_ = new int[total_size_];
  data_ = new double[total_size_];

  // Reset the stream for the second pass
  in.clear();
  in.seekg(0, ios::beg);
  // Second pass: Populate data_, target_words_, and offsets_
  int current_offset = 0;
  int current_source_word = 0;
  offsets_[0] = 0;
  int line_count = 0;
  while (getline(in, line)) {
    tokenizer line_tokenizer(line, sep);
    int source_word, target_word;
    double prob;
    int i = 0;
    for (tokenizer::iterator it = line_tokenizer.begin();
         it != line_tokenizer.end(); ++it) {
      if (i == 0) { // Source word
        source_word = source_vocab->GetIndex(*it);
      } else if (i == 1) { // Target word
        target_word = target_vocab->GetIndex(*it);
      } else if (i == 2) { // Probability
        prob = log(atof(it->c_str()));
      }
      ++i;
    }
    assert(i == 3); // Each line should have exactly 3 tokens.
    if (current_source_word != source_word) {
      assert(current_source_word + 1 == source_word);
      ++current_source_word;
      offsets_[current_source_word] = line_count;
    }
    target_words_[line_count] = target_word;
    data_[line_count] = prob;
    ++line_count;
  }
  assert(line_count == total_size_);
  offsets_[source_count_] = total_size_;
  in.close();
  is_dependent_ = false;
  return true;
}

bool PackedTrie::ReadBinary(const string& filename) {
  std::ifstream in(filename.c_str(), ios::binary | ios::in);
  if (!in.good()) {
    return false;
  }
  // Read the size variables:
  in.read((char*) &source_count_, sizeof(int));
  in.read((char*) &target_count_, sizeof(int));
  in.read((char*) &total_size_, sizeof(int));

  // Allocate space for the arrays and read them in:
  offsets_ = new int[source_count_ + 1];
  target_words_ = new int[total_size_];
  data_ = new double[total_size_];
  in.read((char*) offsets_, sizeof(int) * (source_count_ + 1));
  in.read((char*) target_words_, sizeof(int) * total_size_);
  in.read((char*) data_, sizeof(double) * total_size_);

  in.close();
  is_dependent_ = false;
  return true;
}

void PackedTrie::WriteBinary(const string& filename) const {
  std::ofstream out(filename.c_str(), ios::binary | ios::out);

  // First, write the size variables:
  out.write((char*) &source_count_, sizeof(int));
  out.write((char*) &target_count_, sizeof(int));
  out.write((char*) &total_size_, sizeof(int));

  // Write the arrays:
  out.write((char*) offsets_, sizeof(int) * (source_count_ + 1));
  out.write((char*) target_words_, sizeof(int) * total_size_);
  out.write((char*) data_, sizeof(double) * total_size_);
 
  out.close();
}
