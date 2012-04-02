#ifndef _PARALLEL_CORPUS_H_
#define _PARALLEL_CORPUS_H_

// This class defines a parallel corpus divided into document pairs which may or
// may not have a sentence alignment.

#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include <boost/logic/tribool.hpp>

#include "alignment_models/monotonic_aligner.h"
#include "util/vocab.h"

using std::pair;
using std::set;
using std::string;
using std::vector;

using boost::logic::tribool;
using boost::logic::indeterminate;

class ParallelCorpus {
 public:
  typedef vector<int> Sentence;
  typedef vector<Sentence> Document;
  typedef std::pair<Document, Document> DocumentPair;

  // Partial alignments are stored as a grid. For each source/target pair, it's
  // either true, false, or unknown.
  typedef vector<vector<tribool> > PartialAlignment;

  ParallelCorpus(bool use_lowercase) : use_lowercase_(use_lowercase),
    source_stemming_(false), target_stemming_(false), stemming_length_(4) {}
  ~ParallelCorpus() {}

  // Read documents from files. Each file can contain multiple documents, and
  // document boundaries are delimited by blank lines. Both files must contain
  // the same number of documents.
  // Returns false if either file is unreadable.
  bool ReadDocumentPairs(const string& source_file, const string& target_file);
  // Read documents from files and set their alignment to be monotonic. Returns
  // false if either file is unreadable or if the source and target don't have
  // the same number of sentences.
  bool ReadAlignedPairs(const string& source_file, const string& target_file);
  // Read document pairs along with their alignments.
  bool ReadAlignedPairs(const string& source_file, const string& target_file,
      const string& alignment_file);
  // Read document pairs with partial alignments from mechanical turk.
  bool ReadPartiallyAlignedPairs(const string& source_file,
                                 const string& target_file,
                                 const string& alignment_file);
  // Read documents which are known to be parallel, storing each pair as a
  // document.
  bool ReadParallelData(const string& source_file, const string& target_file);

  // Delete all document pairs and alignments (but keep vocabularies)
  void ClearData();

  // Randomly delete a percentage of the sentences from each side of document
  // pair at index, and adjust the alignment accordingly.
  void RandomDeletion(double percentage, int index);
  // Random delete on all document pairs.
  void RandomDeletion(double percentage);

  // Find the precision, recall, and f1 of a baseline which simply aligns along
  // the diagonal.
  void DiagonalBaseline(double* precision, double* recall, double* f1) const;

  // Print some statistics about the corpus to the given stream.
  void PrintStats(std::ostream& out) const;

  // Print the document pair with aligned sentences side-by-side.
  void PrintDocPair(int index, std::ostream& out) const;

  // Print the source/target sentence pair
  void PrintSentencePair(
      const Sentence& source, const Sentence& target, std::ostream& out) const;
  // Print the given sentence to the stream, using the given vocabulary.
  void PrintSentence(
      const Sentence& sentence, const Vocab& vocab, std::ostream& out) const;

  // Access a document pair or its alignment.
  inline const DocumentPair& GetDocPair(int i) const {
    return doc_pairs_.at(i);
  }
  inline const set<pair<int, int> >& GetAlignment(int i) const {
    return alignments_.at(i);
  }
  inline const PartialAlignment& GetPartialAlignment(int i) const {
    return partial_alignments_.at(i);
  }
  inline int size() const { return doc_pairs_.size(); }

  // Vocabulary accessors
  inline const Vocab& source_vocab() const { return source_vocab_; }
  inline const Vocab& target_vocab() const { return target_vocab_; }

  // Adds entries to the source and target vocabularies
  void AddSourceVocab(const Vocab& v);
  void AddTargetVocab(const Vocab& v);

  // Accessors for the stemming variables
  void SetSourceStemming(bool source_stemming) {
    source_stemming_ = source_stemming;
  }
  void SetTargetStemming(bool target_stemming) {
    target_stemming_ = target_stemming;
  }
  bool source_stemming() const { return source_stemming_; }
  bool target_stemming() const { return target_stemming_; }
  void SetStemmingLength(int stemming_length);
  int stemming_length() const { return stemming_length_; }

 private:
  // Reads sentences from this stream into documents. Either source_vocab_ or
  // target_vocab_ is passed to this.
  void ReadDocuments(std::ifstream* in, vector<Document>* docs, Vocab* vocab,
      bool use_stemming);
  // Read alignments from a file, and add them to alignments_. Returns false if
  // unless alignments_ is the same size as doc_pairs_ after all alignments are
  // read.
  bool ReadAlignmentFile(const string& filename);
  // This also fills alignments_ with the known alignments in addition to
  // populating partial_alignments_.
  bool ReadPartialAlignmentFile(const string& filename);

  // Initialize a partial alignment grid with "indeterminant" for all sentence
  // pairs.
  void InitPartialAlignment(const DocumentPair& doc_pair,
                            PartialAlignment* partial_alignment) const;

  inline void Stem(string& word) const {
    word.resize(stemming_length_);
  }

  // If true, the words will be lowercased before being indexed by the
  // vocabulary.
  bool use_lowercase_;
  // Apply stemming to source/target words. These are false by default.
  bool source_stemming_, target_stemming_;
  int stemming_length_;

  // Source and target documents, as well as a possible alignment between them.
  vector<DocumentPair> doc_pairs_;
  vector<set<pair<int, int> > > alignments_;
  vector<PartialAlignment> partial_alignments_;
  // Vocabularies for the source and target docs.
  Vocab source_vocab_;
  Vocab target_vocab_;
};

#endif
