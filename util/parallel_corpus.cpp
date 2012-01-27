#include "util/parallel_corpus.h"

#include <cstdlib>
#include <string>
#include <utility>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>

bool ParallelCorpus::ReadDocumentPairs(const string& source_file,
                                       const string& target_file) {
  vector<Document> source_docs;
  std::ifstream source_in(source_file.c_str());
  if (source_in) {
    ReadDocuments(&source_in, &(source_docs), &source_vocab_, source_stemming_);
  } else {
    return false;
  }
  source_in.close();

  vector<Document> target_docs;
  std::ifstream target_in(target_file.c_str());
  if (target_in) {
    ReadDocuments(&target_in, &(target_docs), &target_vocab_, target_stemming_);
  } else {
    return false;
  }
  target_in.close();
  if (source_docs.size() != target_docs.size()) {
    return false;
  }
  for (int i = 0; i < source_docs.size(); ++i) {
    DocumentPair doc_pair;
    doc_pair.first.swap(source_docs.at(i));
    doc_pair.second.swap(target_docs.at(i));
    doc_pairs_.push_back(doc_pair);
  }
  return true;
}

bool ParallelCorpus::ReadAlignedPairs(const string& source_file,
                                      const string& target_file) {
  int old_doc_size = doc_pairs_.size();
  if (ReadDocumentPairs(source_file, target_file)) {
    // Check to see if the document that was just read has the same number of
    // source and target sentences.
    for (int i = old_doc_size; i < doc_pairs_.size(); ++i) {
      if (doc_pairs_.at(i).first.size() == doc_pairs_.at(i).second.size()) {
        set<pair<int, int> > al;
        for (int j = 0; j < doc_pairs_.at(i).first.size(); ++j) {
          al.insert(std::make_pair(j, j));
        }
        alignments_.resize(i + 1);
        alignments_[i] = al;
      } else {
        return false;
      }
    }
    return true;
  } 
  return false;
}

bool ParallelCorpus::ReadAlignedPairs(const string& source_file,
                                      const string& target_file,
                                      const string& alignment_file) {
  int old_doc_size = doc_pairs_.size();
  if (ReadDocumentPairs(source_file, target_file)) {
    if (ReadAlignmentFile(alignment_file)) {
      return true;
    }
  }
  return false;
}

bool ParallelCorpus::ReadParallelData(const string& source_file,
                                      const string& target_file) {
  typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
  boost::char_separator<char> sep(" \t");
  std::string line;

  vector<Sentence> source_sents;
  std::ifstream source_in(source_file.c_str());
  if (source_in.good()) {
    Document doc;
    while (getline(source_in, line)) {
      Sentence current_sentence;
      tokenizer line_tokenizer(line, sep);
      for (tokenizer::iterator it = line_tokenizer.begin();
           it != line_tokenizer.end(); ++it) {
        string token = *it;
        if (use_lowercase_) {
          boost::to_lower(token);
        }
        if (source_stemming_) {
          Stem(token);
        }
        current_sentence.push_back(source_vocab_.AddWord(token));
      }
      source_sents.push_back(current_sentence);
    }
    source_in.close();
  } else {
    return false;
  }

  vector<Sentence> target_sents;
  std::ifstream target_in(target_file.c_str());
  if (target_in.good()) {
    Document doc;
    while (getline(target_in, line)) {
      Sentence current_sentence;
      tokenizer line_tokenizer(line, sep);
      for (tokenizer::iterator it = line_tokenizer.begin();
           it != line_tokenizer.end(); ++it) {
        string token = *it;
        if (use_lowercase_) {
          boost::to_lower(token);
        }
        if (target_stemming_) {
          Stem(token);
        }
        current_sentence.push_back(target_vocab_.AddWord(token));
      }
      target_sents.push_back(current_sentence);
    }
    target_in.close();
  } else {
    return false;
  }
  if (source_sents.size() != target_sents.size()) {
    return false;
  }
  for (int i = 0; i < source_sents.size(); ++i) {
    //if ((source_sents.at(i).size() > 0)
    //  && (target_sents.at(i).size() > 0)) {
      DocumentPair doc_pair;
      doc_pair.first.push_back(source_sents.at(i));
      doc_pair.second.push_back(target_sents.at(i));
      doc_pairs_.push_back(doc_pair);
    //}
  }
  return true;
}

void ParallelCorpus::ClearData() {
  doc_pairs_.clear();
  alignments_.clear();
}

void ParallelCorpus::RandomDeletion(double percentage, int index) {
  //std::cout << "Before:" << std::endl;
  //PrintPair(index, std::cout);
  // TODO: Currently Broken by alignment change
  Document& source = doc_pairs_.at(index).first;
  Document& target = doc_pairs_.at(index).second;
  set<pair<int, int> >& alignment = alignments_.at(index);

  int deletions = (int) (percentage * source.size());
  vector<int> removed_source;
  while (deletions > 0) {
    int s = rand() % source.size();
    //std::cout << "Deleting source sentence " << s << std::endl;
    source.erase(source.begin() + s);
    removed_source.push_back(s);
    deletions--;
  }
  deletions = (int) (percentage * target.size());
  vector<int> removed_target;
  while (deletions > 0) {
    int t = rand() % target.size();
    //std::cout << "Deleting target sentence " << t << std::endl;
    target.erase(target.begin() + t);
    removed_target.push_back(t);
    deletions--;
  }
  for (int i = 0; i < removed_source.size(); ++i) {
    set<pair<int, int> >::iterator it;
    for (it = alignment.begin(); it != alignment.end(); ) {
      if (it->first == removed_source.at(i)) {
        set<pair<int, int> >::iterator temp = it;
        ++it;
        alignment.erase(temp);
      } else {
        ++it;
      }
    }
    /*
    int rs = removed_source.at(i);
    int s = 0;
    int t = 0;
    for (int a = 0; a < alignment.size(); ++a) {
      if (s == rs) {
        switch (alignment.at(a)) {
          case ALIGNOP_DELETE: // 1:0
            alignment.erase(alignment.begin() + a);
            a = alignment.size();
            break;
          case ALIGNOP_INSERT: // 0:1
            Alignment::UpdatePositions(alignment.at(a), false, &s, &t);
            break;
          case ALIGNOP_MATCH: // 1:1
            alignment[a] = ALIGNOP_INSERT;
            a = alignment.size();
            break;
          default:
            std::cerr << "Not yet implemented" << std::endl;
            exit(-1);
        }
      } else {
        Alignment::UpdatePositions(alignment.at(a), false, &s, &t);
      }
    }
    */
  }
  for (int i = 0; i < removed_target.size(); ++i) {
    set<pair<int, int> >::iterator it;
    for (it = alignment.begin(); it != alignment.end(); ) {
      if (it->second == removed_target.at(i)) {
        set<pair<int, int> >::iterator temp = it;
        ++it;
        alignment.erase(temp);
      } else {
        ++it;
      }
    }
    /*
    int rt = removed_target.at(i);
    int s = 0;
    int t = 0;
    for (int a = 0; a < alignment.size(); ++a) {
      if (t == rt) {
        switch (alignment.at(a)) {
          case ALIGNOP_DELETE: // 1:0
            Alignment::UpdatePositions(alignment.at(a), false, &s, &t);
            break;
          case ALIGNOP_INSERT: // 0:1
            alignment.erase(alignment.begin() + a);
            a = alignment.size();
            break;
          case ALIGNOP_MATCH: // 1:1
            alignment[a] = ALIGNOP_DELETE;
            a = alignment.size();
            break;
          default:
            std::cerr << "Not yet implemented" << std::endl;
            exit(-1);
        }
      } else {
        Alignment::UpdatePositions(alignment.at(a), false, &s, &t);
      }
    }
    */
  }
  //std::cout << "After:" << std::endl;
  //PrintPair(index, std::cout);
}

void ParallelCorpus::RandomDeletion(double percentage) {
  for (int i = 0; i < doc_pairs_.size(); ++i) {
    RandomDeletion(percentage, i);
  }
}

void ParallelCorpus::DiagonalBaseline(
    double* precision, double* recall, double* f1) const {
  double true_positives = 0.0;
  double proposed_positives = 0.0;
  double total_positives = 0.0;
  for (int i = 0; i < doc_pairs_.size(); ++i) {
    int source_size = doc_pairs_.at(i).first.size();
    int target_size = doc_pairs_.at(i).second.size();
    vector<AlignmentOperation> diagonal_alignment;
    while (diagonal_alignment.size() < std::min(source_size, target_size)) {
      diagonal_alignment.push_back(ALIGNOP_MATCH);
    }
    // This alignment is possibly incomplete, but for now that doesn't
    // matter for evaluation.
    Alignment::CompareAlignments(alignments_.at(i), diagonal_alignment,
        &true_positives, &proposed_positives, &total_positives);
  }
  *precision = true_positives / proposed_positives;
  *recall = true_positives / total_positives;
  *f1 = 2 * (((*precision) * (*recall)) / ((*precision) + (*recall)));
}

void ParallelCorpus::PrintStats(std::ostream& out) const {
  int source_sentences = 0;
  int target_sentences = 0;
  int sentence_pairs = 0;
  int parallel_sentences = 0;
  int source_words = 0;
  int target_words = 0;
  int word_pairs = 0;
  // TODO: Fix
  /*
  for (int i = 0; i < doc_pairs_.size(); ++i) {
    const Document& source = doc_pairs_.at(i).first;
    const Document& target = doc_pairs_.at(i).second;
    const vector<AlignmentOperation>& alignment = alignments_.at(i);
    source_sentences += source.size();
    target_sentences += target.size();
    sentence_pairs += source.size() * target.size();
    int current_source_words = 0;
    int current_target_words = 0;
    for (int s = 0; s < source.size(); ++s) {
      current_source_words += source.at(s).size();
    }
    for (int t = 0; t < target.size(); ++t) {
      current_target_words += target.at(t).size();
    }
    for (int a = 0; a < alignment.size(); ++a) {
      if (alignment.at(a) == ALIGNOP_MATCH) {
        parallel_sentences++;
      }
    }
    source_words += current_source_words;
    target_words += current_target_words;
    word_pairs += current_source_words * current_target_words;
  }
  */
  out << "Source sentences: " << source_sentences << "\tTarget sentences: "
      << target_sentences << "\tSentence pairs: " << sentence_pairs
      << "\tParallel sentences: " << parallel_sentences << std::endl;
  out << "Source words: " << source_words << "\tTarget words: "
      << target_words << "\tWord pairs: " << word_pairs << std::endl;
}

void ParallelCorpus::PrintDocPair(int index, std::ostream& out) const {
  const Document& source = doc_pairs_.at(index).first;
  const Document& target = doc_pairs_.at(index).second;
  // TODO
  /*
  const vector<AlignmentOperation>& alignment = alignments_.at(index);
  int s = 0;
  int t = 0;
  for (int a = 0; a < alignment.size(); ++a) {
    switch(alignment.at(a)) {
      case ALIGNOP_DELETE: // 1:0
        out << "(" << s << ") ";
        PrintSentence(source.at(s), source_vocab_, out);
        out << "\t---" << std::endl;
        break;
      case ALIGNOP_INSERT: // 0:1
        out << "---\t";
        out << "(" << t << ") ";
        PrintSentence(target.at(t), target_vocab_, out);
        out << std::endl;
        break;
      case ALIGNOP_MATCH: // 1:1
        out << "(" << s << ") ";
        PrintSentence(source.at(s), source_vocab_, out);
        out << "\t";
        out << "(" << t << ") ";
        PrintSentence(target.at(t), target_vocab_, out);
        out << std::endl;
        break;
      default:
        std::cerr << "Not yet implemented" << std::endl;
        exit(-1);
    }
    Alignment::UpdatePositions(alignment.at(a), false, &s, &t);
  }
  */
}

void ParallelCorpus::PrintSentencePair(
    const Sentence& source, const Sentence& target, std::ostream& out) const {
  PrintSentence(source, source_vocab_, out);
  out << std::endl;
  PrintSentence(target, target_vocab_, out);
  out << std::endl;
}

void ParallelCorpus::PrintSentence(
    const Sentence& sentence, const Vocab& vocab, std::ostream& out) const {
  if (sentence.size() > 0) {
    out << vocab.GetWord(sentence.at(0));
  }
  for (int i = 1; i < sentence.size(); ++i) {
    out << " " << vocab.GetWord(sentence.at(i));
  }
}

void ParallelCorpus::AddSourceVocab(const Vocab& v) {
  source_vocab_.Merge(v);
}

void ParallelCorpus::AddTargetVocab(const Vocab& v) {
  target_vocab_.Merge(v);
}

void ParallelCorpus::ReadDocuments(std::ifstream* in,
                                   vector<Document>* docs,
                                   Vocab* vocab,
                                   bool use_stemming) {
  typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
  boost::char_separator<char> sep(" \t");
  std::string line;

  Document doc;
  while (getline(*in, line)) {
    Sentence current_sentence;
    tokenizer line_tokenizer(line, sep);
    for (tokenizer::iterator it = line_tokenizer.begin();
         it != line_tokenizer.end(); ++it) {
      string token = *it;
      if (use_lowercase_) {
        boost::to_lower(token);
      }
      if (use_stemming) {
        Stem(token);
      }
      current_sentence.push_back(vocab->AddWord(token));
    }
    if (current_sentence.size() > 0) {
      doc.push_back(current_sentence);
    } else {
      // An empty line indicates a document boundary
      if (doc.size() > 0) {
        docs->push_back(doc);
        doc.clear();
      }
    }
  }
  if (doc.size() > 0) {
    docs->push_back(doc);
  }
}

bool ParallelCorpus::ReadAlignmentFile(const string& filename) {
  std::ifstream in(filename.c_str());
  if (!in.good()) {
    return false;
  }

  typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
  boost::char_separator<char> sep(" \t");
  std::string line;

  set<pair<int, int> > alignment;
  while (getline(in, line)) {
    vector<std::string> tokens;
    tokenizer line_tokenizer(line, sep);
    for (tokenizer::iterator it = line_tokenizer.begin();
         it != line_tokenizer.end(); ++it) {
      string token = *it;
      tokens.push_back(token);
    }
    if (tokens.size() == 2) {
      int i = atoi(tokens[0].c_str());
      int j = atoi(tokens[1].c_str());
      alignment.insert(std::make_pair(i, j));
    } else {
      // An empty line indicates a document boundary
      if (alignment.size() > 0) {
        alignments_.push_back(alignment);
        alignment.clear();
      }
    }
  }
  if (alignment.size() > 0) {
    alignments_.push_back(alignment);
  }
  in.close();
  if (alignments_.size() != doc_pairs_.size()) {
    return false;
  }
  return true;

}

void ParallelCorpus::SetStemmingLength(int stemming_length) {
  assert(stemming_length > 0);
  stemming_length_ = stemming_length;
}
