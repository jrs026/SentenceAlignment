#ifndef _DOCUMENT_ALIGNER_H_
#define _DOCUMENT_ALIGNER_H_

// This class includes a model which finds a monotonic alignment on sentences in
// a document by using a word alignment model to find the probability that two
// sentences are parallel. This model can also learn the parameters of the word
// alignment model.

#include <iostream>
#include <vector>

#include "boost/multi_array.hpp"
#include "boost/array.hpp"
#include "alignment_models/model1.h"
#include "alignment_models/monotonic_aligner.h"
#include "util/nullbuf.h"
#include "util/parallel_corpus.h"

using std::vector;

// This score is used to identify elements in the cache that are empty.
// Since these are log probabilities, no valid value will be positive.
#define UNDEFINED_SCORE 1.0

template<uint8_t order>
class DocumentAligner;

template<uint8_t order>
class AlignedDocumentPair : public SequencePair<order> {
 friend class DocumentAligner<order>;

 private:
  typedef typename SequencePair<order>::AlignmentState AlignmentState;
  typedef ParallelCorpus::DocumentPair DocumentPair;
  typedef ParallelCorpus::Sentence Sentence;

 public:
  AlignedDocumentPair() {}
  virtual ~AlignedDocumentPair() {
    delete posteriors_;
    delete cached_pair_scores_;
    delete source_costs_;
    delete target_costs_;
  }

  // Factory function. The objects passed in must exist for the
  // lifetime of this aligned document pair, and the aligned document pair will
  // own the source/target cost vectors.
  static AlignedDocumentPair<order>* CreateDocumentPair(
      const DocumentPair* doc_pair, double alignment_prior,
      vector<double>* source_costs, vector<double>* target_costs,
      Model1* model1);

  // Inherited functions.
  virtual double GetScore(const AlignmentState& state,
                          const AlignmentOperation align_op) const;
  virtual double GetScoreAndUpdate(const AlignmentState& state,
                                   const AlignmentOperation align_op,
                                   const double expected_prob);
  virtual void ObservedArcUpdate(const AlignmentState& state,
                                 const AlignmentOperation align_op);

  // Update the expected counts of the Model1 object after ForwardBackward has
  // been run on this pair.
  virtual void UpdateExpectedCounts(double likelihood);

  // Clear the cached Model 1 scores.
  virtual void ClearCachedScores();
  virtual void SetLambda(double lambda) {
    alignment_prior_ = lambda;
  }

 private:
  const DocumentPair* doc_pair_;
  double alignment_prior_;
  // Source and target language model costs.
  vector<double>* source_costs_;
  vector<double>* target_costs_;
  Model1* model1_;
  // Posterior probabilities of each sentence pair being aligned.
  boost::multi_array<double, 2>* posteriors_;
  // Cached Model 1 scores for each sentence pair.
  boost::multi_array<double, 2>* cached_pair_scores_;
};

template<uint8_t order>
class DocumentAligner {
 private:
  typedef ParallelCorpus::DocumentPair DocumentPair;
  typedef ParallelCorpus::Sentence Sentence;
 public:
  // Arguments:
  //   pc : A parallel corpus which is the model's observed data
  //   alignment_prior : A model parameter which acts as a prior for how likely
  //                     it is for sentences to be aligned (between 0 and 1)
  //   alpha : The symmetric Dirichlet prior for Model 1's translation
  //           probabilities (must be greater than 0, and >= 1 if regular EM is
  //           used.
  //   use_poisson_lm : When this is true, the language models will directly
  //                    predict the length of sentences through the Poisson
  //                    distribution, rather than predicting the EOS.
  DocumentAligner(const ParallelCorpus* pc, double alignment_prior,
      double alpha, bool use_poisson_lm); 
  ~DocumentAligner() {
    delete aligner_;
    for (int i = 0; i < aligned_pairs_.size(); ++i) {
      delete aligned_pairs_[i];
    }
  }

  // Run an EM iteration on the parallel corpus, returning the log likelihood.
  // If variational is true, variational Bayes will be used.
  double EM(bool variational, int max = 0);

  // Check the accuracy of the aligner on the parallel corpus. Only look at the
  // first (max) document pairs.
  void Test(int max, double* precision, double* recall, double* f1,
      std::ostream& out = cnull);

  // Access the underlying word alignment model.
  const Model1& GetModel1() const {
    return model1_;
  }
  Model1* MutableModel1() {
    return &model1_;
  }

  void SetLambda(double lambda) {
    alignment_prior_ = lambda;
    for (int i = 0; i < aligned_pairs_.size(); ++i) {
      aligned_pairs_[i]->SetLambda(lambda);
    }
  }

 private:
  void CreateLanguageModels();
  double SourceLMCost(const Sentence& sentence) const;
  double TargetLMCost(const Sentence& sentence) const;

  // Data that the model will run on
  const ParallelCorpus* pc_;
  // Stored aligned pairs which are used to save state between EM iterations.
  vector<AlignedDocumentPair<order>* > aligned_pairs_;
  // Model objects
  Model1 model1_;
  MonotonicAligner<order>* aligner_;
  // This value adjusts the tradeoff between the substitution arc and the
  // insertion/deletion arcs
  double alignment_prior_;
  // Language model costs
  vector<double> source_lm_, target_lm_;
  // Source/target average sentence lengths
  double source_length_, target_length_;
  // When this is true, the language models will directly predict the length of
  // sentences through the Poisson distribution, rather than predicting the EOS.
  const bool use_poisson_lm_;
};

#endif
