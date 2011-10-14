#ifndef _PARALLEL_CORPUS_H_
#define _PARALLEL_CORPUS_H_

// This class defines a parallel corpus divided into document pairs which may or
// may not have a sentence alignment.

#include <fstream>
#include <vector>

#include "alignment_models/monotonic_aligner.h"
#include "util/vocab.h"

class ParallelCorpus {
 public:
  typedef std::vector<int> Sentence;
  typedef std::vector<Sentence> Document;
  typedef std::pair<Document, Document> DocumentPair;
  typedef std::vector<AlignmentOperation> Alignment;

  ParallelCorpus(bool use_lowercase) : use_lowercase_(use_lowercase) {}
  ~ParallelCorpus() {}

  // Read two documents from files.
  // Returns false if either file is unreadable.
  bool ReadDocumentPair(const string& source_file, const string& target_file);
  // Read documents from files and set their alignment to be monotonic. Returns
  // false if either file is unreadable or if the source and target don't have
  // the same number of sentences.
  bool ReadAlignedPair(const string& source_file, const string& target_file);

  // Access a document pair or its alignment.
  inline const DocumentPair& GetDocPair(int i) const {
    return doc_pairs_.at(i);
  }
  inline const Alignment& GetAlignment(int i) const {
    return alignments_.at(i);
  }
  inline int size() const { return doc_pairs_.size(); }

  // Vocabulary accessors
  inline const Vocab& source_vocab() const { return source_vocab_; }
  inline const Vocab& target_vocab() const { return target_vocab_; }

 private:
  // Reads sentences from this stream into a document. Either source_vocab_ or
  // target_vocab_ is passed to this.
  void ReadDocument(std::ifstream* in, Document* doc, Vocab* vocab);
  // If true, the words will be lowercased before being indexed by the
  // vocabulary.
  bool use_lowercase_;
  // Source and target documents, as well as a possible alignment between them.
  vector<DocumentPair> doc_pairs_;
  vector<Alignment> alignments_;
  // Vocabularies for the source and target docs.
  Vocab source_vocab_;
  Vocab target_vocab_;
};

#endif
