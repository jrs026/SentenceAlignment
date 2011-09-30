#include "alignment_models/monotonic_aligner.h"

#include <algorithm>

#include "util/math_util.h"

using std::vector;
using std::numeric_limits;

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
    SequencePair* labeled_pair) const {
  AlignmentState current_node, next_node;
  for (int i = 0; i < order + 2; ++i) {
    current_node[i] = 0;
  }
  const vector<AlignmentOperation>& alignment = labeled_pair.alignment();
  for (int i = 0; i < alignment.size(); ++i) {
    labeled_pair->ObservedArcUpdate(current_node, alignment.at(i));
    GetSinkState(current_node, alignment.at(i), &next_node);
    current_node.swap(next_node);
  }
}

template<uint8_t order>
double MonotonicAligner<order>::ForwardBackward(
    SequencePair* sequence_pair) const {
  AlignmentLattice* alphas;
  CreateAlignmentLattice(*sequence_pair, alphas);
  AlignmentState source_state, sink_state;
  // Alpha pass
  // Initialize the start state with score 0.
  for (int i = 0; i < order + 2; ++i) {
    sink_state[i] = 0;
  }
  (*alphas)(sink_state) = 0.0;
  double Z = -numeric_limits<double>::max();
  // Update the rest of the states by their incoming arcs.
  while (IncrementState(&source_state)) {
    if (!IsValid(sink_state)) {
      continue;
    }
    double value = -numeric_limits<double>::max();
    for (AlignmentOperation i = 0; i < max_alignment_op_; ++i) {
      GetSourceState(sink_state, i, &source_state);
      if (IsValid(source_state)) {
        double arc_score = (*alphas)(source_state);
        if (arc_score > -numeric_limits<double>::max()) {
          arc_score += GetScore(source_state, i);
          value = MathUtil::LogAdd(value, arc_score); 
        }
      }
    }
    (*alphas)(sink_state) = value;
    if ((sink_state[0] == sequence_pair->GetInputLength())
        && (sink_state[1] == sequence_pair->GetOutputLength())) {
      Z = MathUtil::LogAdd(Z, value);
    }
  }
  std::cout << "Alpha pass sum: " << exp(Z) << std::endl;

  AlignmentLattice* betas;
  CreateAlignmentLattice(*sequence_pair, betas);
  // Beta pass
  source_state[0] = sequence_pair->GetInputLength();
  source_state[1] = sequence_pair->GetOutputLength();
  for (int i = 2; i < order + 2; ++i) {
    source_state[i] = max_alignment_op_ - 1;
  }
  do {
    if (!IsValidState(source_state)) {
      continue;
    }
    // Check to see if we're in a final state, and set the beta value
    // accordingly.
    if ((source_state[0] == sequence_pair->GetInputLength())
        && (source_state[1] == sequence_pair->GetOutputLength())) {
      (*betas)(source_state) = 0.0;
    } else {
      double value = -numeric_limits<double>::max();
      double source_alpha = (*alphas)(source_state);
      for (AlignmentOperation i = 0; i < max_alignment_op_; ++i) {
        GetSinkState(source_state, i, &sink_state);
        if (IsValid(sink_state)) {
          double arc_score = (*betas)(sink_state);
          if (arc_score > -numeric_limits<double>::max()) {
            double prob = arc_score + source_alpha - Z;
            arc_score += GetScoreAndUpdate(source_state, i, prob);
            value = MathUtil::LogAdd(value, arc_score); 
          }
        }
      }
      (*betas)(source_state) = value;
    }
  } while (DecrementState(&source_state));
  for (int i = 0; i < order + 2; ++i) {
    sink_state[i] = 0;
  }
  std::cout << "Beta pass sum: " << exp((*betas)(sink_state)) << std::endl;
  delete alphas, betas;
  return Z;
}

template<uint8_t order>
double MonotonicAligner<order>::Align(SequencePair* sequence_pair) const {
  AlignmentLattice* best_score;
  CreateAlignmentLattice(*sequence_pair, best_score);
  AlignmentState source_state, sink_state;
  // Initialize the start state with score 0.
  for (int i = 0; i < order + 2; ++i) {
    sink_state[i] = 0;
  }
  (*best_score)(sink_state) = 0.0;
  double best_total_score = -numeric_limits<double>::max();
  AlignmentState best_final_state;
  // Update the rest of the states by their incoming arcs.
  while (IncrementState(&source_state)) {
    if (!IsValid(sink_state)) {
      continue;
    }
    double value = -numeric_limits<double>::max();
    for (AlignmentOperation i = 0; i < max_alignment_op_; ++i) {
      GetSourceState(sink_state, i, &source_state);
      if (IsValid(source_state)) {
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
      if (best_total_score < value) {
        best_total_score = value;
        best_final_state = sink_state;
      }
    }
  }

  vector<AlignmentOperation> alignment;
  // Retrace the best path
  sink_state = best_final_state;
  while ((sink_state[0] > 0) || (sink_state[1] > 0)) {
    int old_alignment_size = alignment.size();
    for (AlignmentOperation i = 0; i < max_alignment_op_; ++i) {
      GetSourceState(sink_state, i, &source_state);
      if (IsValid(source_state) 
          || (MathUtil::ApproxEqual(best_score(sink_state),
                (*best_score)(source_state)
                + sequence_pair->GetScore(source_state, i)))) {
        sink_state = source_state;
        alignment.push_back(i);
        break;
      }
    }
    if (alignment.size() == old_alignment_size) {
      std::cerr << "Error while backtracing the alignment" << std::endl;
      exit(-1);
    }
  }
  std::reverse(alignment.begin(), alignment.end());
  return best_total_score;
}

template<uint8_t order>
void MonotonicAligner<order>::CreateAlignmentLattice(
    const SequencePair& sequence_pair,
    AlignmentLattice* lattice) const {
  AlignmentState shape;
  shape.at(0) = sequence_pair.GetInputLength() + 1;
  shape.at(1) = sequence_pair.GetOutputLength() + 1;
  for (int i = 2; i < order + 2; ++i) {
    shape.at(i) = max_alignment_op_;
  }
  lattice = new AlignmentLattice(shape);
  return lattice;
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
  MonotonicAligner<order>::UpdatePositions(
      align_op, false, (*sink)[0], (*sink)[1]);
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
    MonotonicAligner<order>::UpdatePositions(
        sink[order+1], true, (*source)[0], (*source)[1]);
  } else {
    MonotonicAligner<order>::UpdatePositions(
        align_op, true, (*source)[0], (*source)[1]);
  }
}

template<uint8_t order>
bool MonotonicAligner<order>::IsValid(const AlignmentState& node,
                                      const SequencePair& sequence_pair) const {
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
      MonotonicAligner<order>::UpdatePositions(
          node[i], true, &input_pos, &output_pos);
      if ((input_pos < 0) || (output_pos < 0)) {
        return false;
      }
    }
  }
  return true;
}
