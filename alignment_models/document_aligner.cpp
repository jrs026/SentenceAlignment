#include "alignment_models/document_aligner.h"

#include <cstdlib>
#include <limits>

#include "alignment_models/packed_trie.h"

using std::endl;
using std::vector;

// AlignedDocumentPair

template<uint8_t order>
AlignedDocumentPair<order>* AlignedDocumentPair<order>::CreateDocumentPair(
    const DocumentPair* doc_pair,
    double alignment_prior,
    vector<double>* source_costs,
    vector<double>* target_costs,
    Model1* model1) {
  AlignedDocumentPair<order>* aligned_pair = new AlignedDocumentPair<order>();
  aligned_pair->doc_pair_ = doc_pair;
  aligned_pair->alignment_prior_ = alignment_prior;
  aligned_pair->source_costs_ = source_costs;
  aligned_pair->target_costs_ = target_costs;
  aligned_pair->model1_ = model1;
  aligned_pair->input_length_ = doc_pair->first.size();
  aligned_pair->output_length_ = doc_pair->second.size();
  boost::array<boost::multi_array<double, 2>::index, 2> shape = {{
      doc_pair->first.size(), doc_pair->second.size() }};
  aligned_pair->posteriors_ = new boost::multi_array<double, 2>(shape);
  aligned_pair->cached_pair_scores_ = new boost::multi_array<double, 2>(shape);
  aligned_pair->ClearCachedScores();
  return aligned_pair;
}

template<uint8_t order>
double AlignedDocumentPair<order>::GetScore(
    const AlignmentState& state,
    const AlignmentOperation align_op) const {
  double cost = -std::numeric_limits<double>::max();
  int i1, i2, o1, o2;
  SpanFromAlignmentArc(state, align_op, &i1, &i2, &o1, &o2);
  if ((i2 - i1 == 1) && (o2 - o1 == 1)) {
    // Substitution
    if ((*cached_pair_scores_)[i1][o1] == UNDEFINED_SCORE) {
      cost = log(alignment_prior_) + source_costs_->at(i1)
          + LengthCost(doc_pair_->first.at(i1).size(),
                       doc_pair_->second.at(o1).size())
          + model1_->ScorePair(doc_pair_->first.at(i1), doc_pair_->second.at(o1));
      (*cached_pair_scores_)[i1][o1] = cost;
    } else {
      cost = (*cached_pair_scores_)[i1][o1];
    }
    /*
    std::cout << "(" << i1 << ", " << o1 << ") "
        << "Source cost: " << source_costs_->at(i1)
        << " Target cost: " << target_costs_->at(o1)
        << " Pair cost (length): " << 
              log(math_util::Poisson(doc_pair_->first.at(i1).size(),
                                     doc_pair_->second.at(o1).size()))
        << " Pair cost (Model 1): " <<
            model1_->ScorePair(doc_pair_->first.at(i1), doc_pair_->second.at(o1))
        << std::endl;
    */
    //if (i1 == o1) {
    //  std::cout << "Match: " << cost << std::endl;
    //}
  } else if ((i2 - i1 == 1) && (o2 - o1 == 0)) {
    // Deletion
    cost = log((1.0 - alignment_prior_) / 2) + source_costs_->at(i1);
  } else if ((i2 - i1 == 0) && (o2 - o1 == 1)) {
    // Insertion
    cost = log((1.0 - alignment_prior_) / 2) + target_costs_->at(o1);
  } else {
    std::cerr << "Illegal alignment for string pairs." << std::endl;
    exit(-1);
  }
  return cost;
}

template<uint8_t order>
double AlignedDocumentPair<order>::GetScoreAndUpdate(
    const AlignmentState& state,
    const AlignmentOperation align_op,
    const double expected_prob) {
  double cost = GetScore(state, align_op);
  int i1, i2, o1, o2;
  SpanFromAlignmentArc(state, align_op, &i1, &i2, &o1, &o2);
  if ((i2 - i1 == 1) && (o2 - o1 == 1)) {
    // Substitution
    (*posteriors_)[i1][o1] = expected_prob
        + log(alignment_prior_) + source_costs_->at(i1)
        + LengthCost(doc_pair_->first.at(i1).size(),
                     doc_pair_->second.at(o1).size());
    //std::cout << i1 << " " << o1 << " " << (*posteriors_)[i1][o1] << std::endl;
  }
  return cost;
}

template<uint8_t order>
void AlignedDocumentPair<order>::ObservedArcUpdate(
    const AlignmentState& state,
    const AlignmentOperation align_op) {
  assert(0); // TODO: not implemented
}

template<uint8_t order>
void AlignedDocumentPair<order>::UpdateExpectedCounts(double likelihood) {
  PackedTrie* counts = model1_->mutable_counts();
  const PackedTrie& t_table = model1_->t_table();

  // Since entries in the T-Table are accessed multiple times, remember the
  // index of each entry for re-use.
  int total_source_size = 0;
  for (int i = 0; i < doc_pair_->first.size(); ++i) {
    total_source_size += doc_pair_->first.at(i).size();
  }
  int* indices = new int[total_source_size];

  for (int j = 0; j < doc_pair_->second.size(); ++j) {
    const Sentence& target = doc_pair_->second.at(j);
    for (int t = 0; t < target.size(); ++t) {
      double t_prob = -std::numeric_limits<double>::max();
      int null_index = t_table.FindIndex(0, target[t]);
      int doc_index = 0; // Used for accessing indices[]
      for (int i = 0; i < doc_pair_->first.size(); ++i) {
        const Sentence& source = doc_pair_->first.at(i);
        // This alignment probability includes both the sentence and the word
        // alignment probabilities and the length probability.
        double alignment_prob = (*posteriors_)[i][j] + log(1.0 / target.size());
        t_prob = MathUtil::LogAdd(t_prob,
            t_table.Data(null_index) + alignment_prob);
        for (int s = 0; s < source.size(); ++s) {
          indices[doc_index] = t_table.FindIndex(source[s], target[t]);
          t_prob = MathUtil::LogAdd(t_prob,
              t_table.Data(indices[doc_index]) + alignment_prob);
          ++doc_index;
        }
      } 
      // Update the expected counts
      doc_index = 0;
      for (int i = 0; i < doc_pair_->first.size(); ++i) {
        const Sentence& source = doc_pair_->first.at(i);
        double alignment_prob = (*posteriors_)[i][j] + log(1.0 / target.size());
        double null_inc = t_table.Data(null_index) + alignment_prob - t_prob;

        MathUtil::LogPlusEQ(counts->Data(null_index), null_inc);
        for (int s = 0; s < source.size(); ++s) {
          double inc = t_table.Data(indices[doc_index])
              + alignment_prob - t_prob;
          MathUtil::LogPlusEQ(counts->Data(indices[doc_index]), inc);
          ++doc_index;
        }
      }
    }
  }
  delete[] indices;
}

template<uint8_t order>
void AlignedDocumentPair<order>::ClearCachedScores() {
  for (int i = 0; i < doc_pair_->first.size(); ++i) {
    for (int j = 0; j < doc_pair_->second.size(); ++j) {
      (*cached_pair_scores_)[i][j] = UNDEFINED_SCORE;
    }
  }
}

// DocumentAligner

template<uint8_t order>
DocumentAligner<order>::DocumentAligner(const ParallelCorpus* pc,
                                        double alignment_prior,
                                        double alpha,
                                        bool use_poisson_lm) : model1_(alpha),
                                        use_poisson_lm_(use_poisson_lm) {
  pc_ = pc;
  vector<const ParallelCorpus*> pcs;
  pcs.push_back(pc);
  model1_.InitDataStructures(pcs, pc->source_vocab(), pc->target_vocab());
  model1_.ClearExpectedCounts();
  alignment_prior_ = alignment_prior;
  aligner_ = new MonotonicAligner<order>(2);
  CreateLanguageModels();
  for (int i = 0; i < pc_->size(); ++i) {
    const DocumentPair& doc_pair = pc_->GetDocPair(i);
    vector<double>* source_costs = new vector<double>();
    vector<double>* target_costs = new vector<double>();
    for (int s = 0; s < doc_pair.first.size(); ++s) {
      source_costs->push_back(SourceLMCost(doc_pair.first.at(s)));
    }
    for (int t = 0; t < doc_pair.second.size(); ++t) {
      target_costs->push_back(TargetLMCost(doc_pair.second.at(t)));
    }
    AlignedDocumentPair<order>* aligned_pair = 
        AlignedDocumentPair<order>::CreateDocumentPair(
          &doc_pair, alignment_prior_, source_costs, target_costs, &model1_);
    aligned_pairs_.push_back(aligned_pair);
  }
}

template<uint8_t order>
double DocumentAligner<order>::EM(bool variational, int max) {
  assert(max <= aligned_pairs_.size());
  if (max == 0) {
    max = aligned_pairs_.size();
  }
  double total_likelihood = 0.0;
  for (int i = 0; i < max; ++i) {
    double likelihood = aligner_->ForwardBackward(aligned_pairs_.at(i));
//    std::cout << aligned_pairs_.at(i)->doc_pair_->first.size() << " "
//        << aligned_pairs_.at(i)->doc_pair_->second.size() << " "
//        << likelihood << endl;
    aligned_pairs_.at(i)->UpdateExpectedCounts(likelihood);
    total_likelihood += likelihood;
    aligned_pairs_.at(i)->ClearCachedScores();
  }
  model1_.MStep(variational);
  model1_.ClearExpectedCounts();
  return total_likelihood;
}

template<uint8_t order>
void DocumentAligner<order>::Test(int max, double* precision, double* recall,
    double* f1, std::ostream& out) {
  assert(max <= aligned_pairs_.size());
  double t_true_positives = 0.0;
  double t_proposed_positives = 0.0;
  double t_total_positives = 0.0;
  int sentence_pairs = 0;
  for (int i = 0; i < max; ++i) {
    // TODO: Temporary, this will cause too much redundant work in general
    aligned_pairs_.at(i)->ClearCachedScores();

    aligner_->Align(aligned_pairs_.at(i));
    double true_positives = 0.0;
    double proposed_positives = 0.0;
    double total_positives = 0.0;
    Alignment::CompareAlignments(pc_->GetAlignment(i), pc_->GetPartialAlignment(i),
        aligned_pairs_.at(i)->alignment(), &true_positives, &proposed_positives,
        &total_positives);
    t_true_positives += true_positives;
    t_proposed_positives += proposed_positives;
    t_total_positives += total_positives;

    // Debug output, if a stream is provided.
    if (out.good()) {
      set<pair<int, int> > proposed_pairs;
      Alignment::PairsFromAlignmentOps(aligned_pairs_.at(i)->alignment(),
                                       &proposed_pairs);
      set<pair<int, int> >::iterator it;
      while (proposed_pairs.size() > 0) {
        it = proposed_pairs.begin();
        if (pc_->GetAlignment(i).find(*it) != pc_->GetAlignment(i).end()) {
          out << "True Positive: " << endl;
        } else {
          out << "False Positive: " << endl;
        }
        pc_->PrintSentencePair(pc_->GetDocPair(i).first.at(it->first),
                               pc_->GetDocPair(i).second.at(it->second),
                               out);
        out << "Monolingual generation score of source and target: "
            << aligned_pairs_.at(i)->source_costs_->at(it->first)
               + aligned_pairs_.at(i)->target_costs_->at(it->second)
               + (2 * log((1.0 - aligned_pairs_.at(i)->alignment_prior_) / 2)) << endl;
        out << "Bilingual generation score of target given source: "
            << (*(aligned_pairs_.at(i)->cached_pair_scores_))[it->first][it->second]
            << endl;
        out << endl;
        proposed_pairs.erase(it);
      }
    }
  }

  std::cout << "Sentence pairs: " << t_proposed_positives << std::endl;

  *precision = t_true_positives / t_proposed_positives;
  *recall = t_true_positives / t_total_positives;
  *f1 = 2 * (((*precision) * (*recall)) / ((*precision) + (*recall)));
}

template<uint8_t order>
void DocumentAligner<order>::CreateLanguageModels() {
  int source_tokens = 0;
  vector<int> source_counts(pc_->source_vocab().size());
  source_lm_.resize(source_counts.size());
  int target_tokens = 0;
  vector<int> target_counts(pc_->target_vocab().size());
  target_lm_.resize(target_counts.size());
  int source_sentences = 0;
  int target_sentences = 0;
  for (int i = 0; i < pc_->size(); ++i) {
    const DocumentPair& doc_pair = pc_->GetDocPair(i);
    source_sentences += doc_pair.first.size();
    for (int s = 0; s < doc_pair.first.size(); ++s) {
      const Sentence& sentence = doc_pair.first.at(s);
      source_tokens += sentence.size();
      if (!use_poisson_lm_) {
        source_tokens++;
        source_counts[0] += 1; // EOS
      }
      for (int w = 0; w < sentence.size(); ++w) {
        source_counts[sentence.at(w)]++;
      }
    }
    target_sentences += doc_pair.second.size();
    for (int t = 0; t < doc_pair.second.size(); ++t) {
      const Sentence& sentence = doc_pair.second.at(t);
      target_tokens += sentence.size();
      if (!use_poisson_lm_) {
        target_tokens++;
        target_counts[0] += 1; // EOS
      }
      for (int w = 0; w < sentence.size(); ++w) {
        target_counts[sentence.at(w)]++;
      }
    }
  }
  for (int s = 0; s < source_lm_.size(); ++s) {
    if (source_counts.at(s) > 0) {
      source_lm_[s] = log(source_counts.at(s) / (double) source_tokens);
    } else {
      source_lm_[s] = -std::numeric_limits<double>::max();
    }
  }
  for (int t = 0; t < target_lm_.size(); ++t) {
    if (target_counts.at(t) > 0) {
      target_lm_[t] = log(target_counts.at(t) / (double) target_tokens);
    } else {
      target_lm_[t] = -std::numeric_limits<double>::max();
    }
  }
  source_length_ = (double) source_tokens / source_sentences;
  target_length_ = (double) target_tokens / target_sentences;
}

template<uint8_t order>
double DocumentAligner<order>::SourceLMCost(const Sentence& sentence) const {
  double cost;
  if (!use_poisson_lm_) {
    cost = source_lm_.at(0); // EOS cost
  } else {
    cost = log(math_util::Poisson(source_length_, sentence.size()));
  }
  for (int i = 0; i < sentence.size(); ++i) {
    cost += source_lm_.at(sentence.at(i));
  }
  return cost;
}

template<uint8_t order>
double DocumentAligner<order>::TargetLMCost(const Sentence& sentence) const {
  double cost;
  if (!use_poisson_lm_) {
    cost = target_lm_.at(0); // EOS cost
  } else {
    cost = log(math_util::Poisson(target_length_, sentence.size()));
  }
  for (int i = 0; i < sentence.size(); ++i) {
    cost += target_lm_.at(sentence.at(i));
  }
  return cost;
}

template class AlignedDocumentPair<0>;
template class DocumentAligner<0>;
