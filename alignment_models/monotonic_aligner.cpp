#include "alignment_models/monotonic_aligner.h"

#include <algorithm>
#include <utility>

#include "util/math_util.h"

using std::numeric_limits;
using std::pair;
using std::set;
using std::vector;

namespace Alignment {

void PairsFromAlignmentOps(const vector<AlignmentOperation>& alignment,
    set<pair<int, int> >* pairs) {
  int i = 0;
  int j = 0;
  pairs->clear();
  for (int a = 0; a < alignment.size(); ++a) {
    if (alignment.at(a) == ALIGNOP_MATCH) {
      pairs->insert(std::make_pair(i, j));
    }
    UpdatePositions(alignment.at(a), false, &i, &j);
  }
}

void CompareAlignments(const vector<AlignmentOperation>& true_alignment,
    const vector<AlignmentOperation>& proposed_alignment,
    double* true_positives, double* proposed_positives,
    double* total_positives) {
  set<pair<int, int> > true_pairs, proposed_pairs;
  PairsFromAlignmentOps(true_alignment, &true_pairs);
  PairsFromAlignmentOps(proposed_alignment, &proposed_pairs);
  CompareAlignments(true_pairs, proposed_pairs, true_positives,
      proposed_positives, total_positives);
}

void CompareAlignments(const set<pair<int, int> >& true_pairs,
    const PartialAlignment& partial_alignment,
    const vector<AlignmentOperation>& proposed_alignment,
    double* true_positives, double* proposed_positives,
    double* total_positives) {
  set<pair<int, int> > proposed_pairs;
  PairsFromAlignmentOps(proposed_alignment, &proposed_pairs);
  // Remove the unknowns from the proposed pairs
  set<pair<int, int> >::iterator it;
  //int old_size = proposed_pairs.size();
  for (it = proposed_pairs.begin(); it != proposed_pairs.end(); ) {
    if (indeterminate(partial_alignment[it->first][it->second])) {
      proposed_pairs.erase(it++);
    } else {
      ++it;
    }
  }
  //std::cout << "Removed " << old_size - proposed_pairs.size()
  //    << " pairs" << std::endl;

  CompareAlignments(true_pairs, proposed_pairs, true_positives,
      proposed_positives, total_positives);
}

void CompareAlignments(const set<pair<int, int> >& true_pairs,
    const set<pair<int, int> >& proposed_pairs,
    double* true_positives, double* proposed_positives,
    double* total_positives) {
  *proposed_positives += proposed_pairs.size();
  *total_positives += true_pairs.size();
  
  set<pair<int, int> >::const_iterator it;
  for (it = proposed_pairs.begin(); it != proposed_pairs.end(); ++it) {
    if (true_pairs.find(*it) != true_pairs.end()) {
      (*true_positives)++;
    }
  }
}

}  // end namespace Alignment

template<uint8_t order>
MonotonicAligner<order>::MonotonicAligner(uint8_t max_chunk_size)
    : max_chunk_size_(max_chunk_size) {
  if (max_chunk_size_ < 2) {
    std::cerr << "Error: chunk size too small" << std::endl;
    exit(-1);
  }
  // Compute the maximum number of alignment operations from the chunk size,
  // given that 0:0, M:0, and 0:N alignments don't exist for M, N > 1.
  max_alignment_op_ = 3;
  for (int i = 3; i <= max_chunk_size_; ++i) {
    max_alignment_op_ += i - 1;
  }
}

template<uint8_t order>
double MonotonicAligner<order>::ScoreObservedAlignment(
    SequencePair<order>* labeled_pair) const {
  AlignmentState current_node, next_node;
  for (int i = 0; i < order + 2; ++i) {
    current_node[i] = 0;
  }
  const vector<AlignmentOperation>& alignment = labeled_pair->alignment();
  for (int i = 0; i < alignment.size(); ++i) {
    labeled_pair->ObservedArcUpdate(current_node, alignment.at(i));
    GetSinkState(current_node, alignment.at(i), &next_node);
    current_node.swap(next_node);
  }
}

template<uint8_t order>
double MonotonicAligner<order>::ForwardBackward(
    SequencePair<order>* sequence_pair) const {
  AlignmentLattice* alphas = CreateAlignmentLattice(*sequence_pair);
  AlignmentState source_state, sink_state;
  // Alpha pass
  // Initialize the start state with score 0.
  for (int i = 0; i < order + 2; ++i) {
    sink_state[i] = 0;
  }
  double Z = -numeric_limits<double>::max();
  // Update the rest of the states by their incoming arcs.
  do {
    if (!IsValid(sink_state, *sequence_pair)) {
      continue;
    }
    double value = -numeric_limits<double>::max();
    if ((sink_state[0] == 0) && (sink_state[1] == 0)) {
      // The start state has an alpha value of 1 (in the log domain)
      value = 0.0;
    } else {
      for (AlignmentOperation i = 0; i < max_alignment_op_; ++i) {
        GetSourceState(sink_state, i, &source_state);
        if (IsValid(source_state, *sequence_pair)) {
          double arc_score = (*alphas)(source_state);
          if (arc_score > -numeric_limits<double>::max()) {
            arc_score += sequence_pair->GetScore(source_state, i);
            value = MathUtil::LogAdd(value, arc_score); 
          }
        }
      }
    }
    (*alphas)(sink_state) = value;
    if ((sink_state[0] == sequence_pair->GetInputLength())
        && (sink_state[1] == sequence_pair->GetOutputLength())) {
      Z = MathUtil::LogAdd(Z, value);
    }
  } while (IncrementState(*sequence_pair, &sink_state));

  AlignmentLattice* betas = CreateAlignmentLattice(*sequence_pair);
  // Beta pass
  source_state[0] = sequence_pair->GetInputLength();
  source_state[1] = sequence_pair->GetOutputLength();
  for (int i = 2; i < order + 2; ++i) {
    source_state[i] = max_alignment_op_ - 1;
  }
  do {
    if (!IsValid(source_state, *sequence_pair)) {
      continue;
    }
    if ((source_state[0] == sequence_pair->GetInputLength())
        && (source_state[1] == sequence_pair->GetOutputLength())) {
      // The final states have beta values equal to 1 (in the log domain)
      (*betas)(source_state) = 0.0;
    } else {
      double value = -numeric_limits<double>::max();
      double source_alpha = (*alphas)(source_state);
      for (AlignmentOperation i = 0; i < max_alignment_op_; ++i) {
        GetSinkState(source_state, i, &sink_state);
        if (IsValid(sink_state, *sequence_pair)) {
          double arc_score = (*betas)(sink_state);
          if (arc_score > -numeric_limits<double>::max()) {
            double prob = arc_score + source_alpha - Z;
            arc_score += 
                sequence_pair->GetScoreAndUpdate(source_state, i, prob);
            value = MathUtil::LogAdd(value, arc_score); 
          }
        }
      }
      (*betas)(source_state) = value;
    }
  } while (DecrementState(*sequence_pair, &source_state));
  for (int i = 0; i < order + 2; ++i) {
    sink_state[i] = 0;
  }
  /*
  std::cout << "Alpha pass sum: " << Z << "\t"
      << "Beta pass sum: " << (*betas)(sink_state) << std::endl;
  */
  delete alphas, betas;
  return Z;
}

template<uint8_t order>
double MonotonicAligner<order>::Align(
    SequencePair<order>* sequence_pair) const {
  AlignmentLattice* best_score = CreateAlignmentLattice(*sequence_pair);
  AlignmentState source_state, sink_state;
  // Initialize the start state with score 0.
  for (int i = 0; i < order + 2; ++i) {
    sink_state[i] = 0;
  }
  (*best_score)(sink_state) = 0.0;
  double best_total_score = -numeric_limits<double>::max();
  AlignmentState best_final_state;
  // Update the rest of the states by their incoming arcs.
  while (IncrementState(*sequence_pair, &sink_state)) {
    if (!IsValid(sink_state, *sequence_pair)) {
      continue;
    }
    double value = -numeric_limits<double>::max();
    for (AlignmentOperation i = 0; i < max_alignment_op_; ++i) {
      GetSourceState(sink_state, i, &source_state);
      if (IsValid(source_state, *sequence_pair)) {
        double arc_score = (*best_score)(source_state);
        if (arc_score > -numeric_limits<double>::max()) {
          arc_score += sequence_pair->GetScore(source_state, i);
          value = std::max(value, arc_score); 
        }
      }
    }
    (*best_score)(sink_state) = value;
    if ((sink_state[0] == sequence_pair->GetInputLength())
        && (sink_state[1] == sequence_pair->GetOutputLength())) {
      if (best_total_score <= value) {
        best_total_score = value;
        best_final_state = sink_state;
      }
    }
  }
  //std::cout << "Best total score: " << best_total_score << std::endl;

  vector<AlignmentOperation> alignment;
  // Retrace the best path
  sink_state = best_final_state;
  while ((sink_state[0] > 0) || (sink_state[1] > 0)) {
    int old_alignment_size = alignment.size();
    for (AlignmentOperation i = 0; i < max_alignment_op_; ++i) {
      GetSourceState(sink_state, i, &source_state);
      if (IsValid(source_state, *sequence_pair) 
          && (MathUtil::ApproxEqual((*best_score)(sink_state),
                (*best_score)(source_state)
                + sequence_pair->GetScore(source_state, i)))) {
        sink_state = source_state;
        alignment.push_back(i);
        break;
      }
    }
    if (alignment.size() == old_alignment_size) {
      std::cerr << "Error while backtracing the alignment" << std::endl;
      assert(alignment.size() != old_alignment_size);
      exit(-1);
    }
  }
  std::reverse(alignment.begin(), alignment.end());
  sequence_pair->set_alignment(alignment);
  return best_total_score;
}

template<uint8_t order>
typename MonotonicAligner<order>::AlignmentLattice*
MonotonicAligner<order>::CreateAlignmentLattice(
    const SequencePair<order>& sequence_pair) const {
  AlignmentState shape;
  shape[0] = sequence_pair.GetInputLength() + 1;
  shape[1] = sequence_pair.GetOutputLength() + 1;
  for (int i = 2; i < order + 2; ++i) {
    shape[i] = max_alignment_op_;
  }
  return new AlignmentLattice(shape);
}

template<uint8_t order>
void MonotonicAligner<order>::GetSinkState(const AlignmentState& source,
                                           const AlignmentOperation align_op,
                                           AlignmentState* sink) const {
  (*sink)[0] = source[0];
  (*sink)[1] = source[1];
  for (int i = 2; i < order + 1; ++i) {
    (*sink)[i] = source[i+1];
  }
  if (order > 0) {
    (*sink)[order+1] = align_op;
  }
  Alignment::UpdatePositions(align_op, false, &((*sink)[0]), &((*sink)[1]));
}

template<uint8_t order>
void MonotonicAligner<order>::GetSourceState(const AlignmentState& sink,
                                             const AlignmentOperation align_op,
                                             AlignmentState* source) const {
  (*source)[0] = sink[0];
  (*source)[1] = sink[1];
  for (int i = 2; i < order + 1; ++i) {
    (*source)[i+1] = sink[i];
  }
  if (order > 0) {
    (*source)[2] = align_op;
    Alignment::UpdatePositions(
        sink[order+1], true, &((*source)[0]), &((*source)[1]));
  } else {
    Alignment::UpdatePositions(
        align_op, true, &((*source)[0]), &((*source)[1]));
  }
}

template<uint8_t order>
bool MonotonicAligner<order>::IsValid(
    const AlignmentState& node,
    const SequencePair<order>& sequence_pair) const {
  int input_pos = node[0];
  int output_pos = node[1];
  if ((input_pos < 0) || (output_pos < 0)
    || (input_pos > sequence_pair.GetInputLength())
    || (output_pos > sequence_pair.GetOutputLength())) {
    return false;
  }
  // Traverse the alignment history backwards and see if it was possible.
  for (int i = order + 1; i >= 2; --i) {
    // If this is position (0,0), the alignment history should be filled with
    // delete (0) operations.
    if ((input_pos > 0) || (output_pos > 0) || (node[i] > 0)) {
      Alignment::UpdatePositions(node[i], true, &input_pos, &output_pos);
      if ((input_pos < 0) || (output_pos < 0)) {
        return false;
      }
    }
  }
  return true;
}

template<uint8_t order>
bool MonotonicAligner<order>::IncrementState(
    const SequencePair<order>& pair,
    AlignmentState* state) const {
  for (int i = order + 1; i >= 0; --i) {
    (*state)[i]++;
    if (i >= 2) {
      // This part of the alignment state references the alignment history
      if ((*state)[i] < max_alignment_op_) {
        return true;
      }
    } else if (i == 1) {
      // Output sequence position
      if ((*state)[i] <= pair.GetOutputLength()) {
        return true;
      }
    } else {
      // Input sequence position
      if ((*state)[i] <= pair.GetInputLength()) {
        return true;
      }
    }

    // If we haven't returned by now, this means we need to carry the addition
    // to the next spot in the state index.
    (*state)[i] = 0;
  }
  // We have carried past all of the indices, so we must be trying to increment
  // past the last state in the lattice.
  return false;
}

template<uint8_t order>
bool MonotonicAligner<order>::DecrementState(
    const SequencePair<order>& pair,
    AlignmentState* state) const {
  for (int i = order + 1; i >= 0; --i) {
    if ((*state)[i] > 0) {
      (*state)[i]--;
      return true;
    } else {
      if (i >= 2) {
        // This part of the alignment state references the alignment history
        (*state)[i] = max_alignment_op_ - 1;
      } else if (i == 1) {
        // Output sequence position
        (*state)[i] = pair.GetOutputLength();
      } else {
        // Input sequence position
        (*state)[i] = pair.GetInputLength();
      }
    }
  }
  return false;
}

template class SequencePair<0>;
template class SequencePair<1>;
template class MonotonicAligner<0>;
template class MonotonicAligner<1>;
