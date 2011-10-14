#include "util/parallel_corpus.h"

#include <string>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>

bool ParallelCorpus::ReadDocumentPair(const string& source_file,
                                      const string& target_file) {
  DocumentPair doc_pair;
  std::ifstream source_in(source_file.c_str());
  if (source_in) {
    ReadDocument(&source_in, &(doc_pair.first), &source_vocab_);
  } else {
    return false;
  }
  source_in.close();

  std::ifstream target_in(target_file.c_str());
  if (target_in) {
    ReadDocument(&target_in, &(doc_pair.second), &target_vocab_);
  } else {
    return false;
  }
  target_in.close();
  doc_pairs_.push_back(doc_pair);
  return true;
}

bool ParallelCorpus::ReadAlignedPair(const string& source_file,
                                     const string& target_file) {
  if (ReadDocumentPair(source_file, target_file)) {
    // Check to see if the document that was just read has the same number of
    // source and target sentences.
    int index = doc_pairs_.size() - 1;
    if (doc_pairs_.at(index).first.size() ==
        doc_pairs_.at(index).second.size()) {
      Alignment al(doc_pairs_.at(index).first.size(), ALIGNOP_MATCH);
      alignments_.resize(index + 1);
      alignments_[index] = al;
      return true;
    } else {
      // Remove the document that was just added.
      doc_pairs_.pop_back();
    }
  } 
  return false;
}

void ParallelCorpus::ReadDocument(std::ifstream* in,
                                  Document* doc,
                                  Vocab* vocab) {
  typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
  boost::char_separator<char> sep(" \t");
  std::string line;

  while (getline(*in, line)) {
    Sentence current_sentence;
    tokenizer line_tokenizer(line, sep);
    for (tokenizer::iterator it = line_tokenizer.begin();
         it != line_tokenizer.end(); ++it) {
      string token = *it;
      if (use_lowercase_) {
        boost::to_lower(token);
      }
      current_sentence.push_back(vocab->AddWord(token));
    }
    doc->push_back(current_sentence);
  }
}
