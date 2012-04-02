#ifndef _MONOTONIC_ALIGNER_H_
#define _MONOTONIC_ALIGNER_H_

// MonotonicAligner.h
//
// A generic alignment model which allows m:n alignments and can incorporate
// some amount of alignment history. The SequencePair class allows different
// kinds of models (discriminative or generative).

#include <cstdlib>
#include <cinttypes>
#include <iostream>
#include <set>
#include <vector>

#include <boost/array.hpp>
#include <boost/multi_array.hpp>
#include <boost/logic/tribool.hpp>

// Some definitions of alignment operations that are used by other classes.
#define ALIGNOP_DELETE 0
#define ALIGNOP_INSERT 1
#define ALIGNOP_MATCH 2 

using std::pair;
using std::set;
using std::vector;

using boost::logic::tribool;
using boost::logic::indeterminate;

// An alignment operation includes the basic insertion/deletion/substitution
// operations of edit distance, as well as m:n operations.
typedef uint16_t AlignmentOperation;

typedef vector<vector<tribool> > PartialAlignment;

namespace Alignment {

  // Update the positions based on the alignment operation taken. If reversed is
  // true, update the positions by going backwards instead of forwards.
  // T must be a numeric type.
  template<class T>
  inline void UpdatePositions(const AlignmentOperation align_op,
      bool reversed, T* input_pos, T* output_pos) {
    int update = (reversed ? -1 : 1);
    switch (align_op) {
      case ALIGNOP_DELETE: // 1:0 (Deletion)
        *input_pos += update;
        break;
      case ALIGNOP_INSERT: // 0:1 (Insertion)
        *output_pos += update;
        break;
      case ALIGNOP_MATCH: // 1:1 (Substitution)
        *input_pos += update;
        *output_pos += update;
        break;
      default:
        std::cerr << "Not yet implemented" << std::endl;
        exit(-1);
    }
  }

  // Convert a sequence of alignment operations into source/target index pairs.
  void PairsFromAlignmentOps(const vector<AlignmentOperation>& alignment,
      set<pair<int, int> >* pairs);
  // Gather the statistics needed to compute precision/recall from the proposed
  // and true alignments.
  void CompareAlignments(const vector<AlignmentOperation>& true_alignment,
      const vector<AlignmentOperation>& proposed_alignment,
      double* true_positives, double* proposed_positives,
      double* total_positives);
  void CompareAlignments(const set<pair<int, int> >& true_pairs,
      const PartialAlignment& partial_alignment,
      const vector<AlignmentOperation>& proposed_alignment,
      double* true_positives, double* proposed_positives,
      double* total_positives);
  void CompareAlignments(const set<pair<int, int> >& true_pairs,
      const set<pair<int, int> >& proposed_pairs,
      double* true_positives, double* proposed_positives,
      double* total_positives);

}  // end namespace

// A sequence pair is the generic object that the monotonic aligner operates on.
template <uint8_t order>
class SequencePair {
 protected:
  typedef boost::array<boost::detail::multi_array::index, order + 2>
      AlignmentState;

 public:
  SequencePair() {}
  virtual ~SequencePair() {}

  // These functions return the lengths of the sequences, and should not
  // change.
  inline int GetInputLength() const { return input_length_; }
  inline int GetOutputLength() const { return output_length_; }

  // Return the score of the arc corresponding to taking the given alignment
  // operation at the given state.
  virtual double GetScore(const AlignmentState& state,
                          const AlignmentOperation align_op) const = 0;

  // Returns the score for aligning the two spans and makes any updates
  // necessary based on the expected probability of passing through this arc
  // (the probability passed through to this function does not include the
  // arc's score itself, since it is computed in this function).
  // This will be called during forward-backward, and can be used in either
  // labeled or unlabeled sequence pairs.
  virtual double GetScoreAndUpdate(const AlignmentState& state,
                                   const AlignmentOperation align_op,
                                   const double expected_prob) = 0;

  // This is called during supervised training on all observed arcs in the true
  // alignment.
  virtual void ObservedArcUpdate(const AlignmentState& state,
                                 const AlignmentOperation align_op) = 0;

  // Accessors for the sequence pair's alignment
  const vector<AlignmentOperation>& alignment() const { return alignment_; }
  void set_alignment(const vector<AlignmentOperation>& alignment) {
    alignment_ = alignment;
  }

  // Returns the input/output spans given and alignment state and alignment
  // operation taken at that state.
  inline void SpanFromAlignmentArc(const AlignmentState& state,
      const AlignmentOperation align_op,
      int* i1, int* i2, int* o1, int* o2) const {
    *i1 = *i2 = state[0];
    *o1 = *o2 = state[1];
    Alignment::UpdatePositions(align_op, false, i2, o2);
  }

 protected:
  // These must be set by the subclass and left unchanged.
  int input_length_;
  int output_length_;

 private:
  vector<AlignmentOperation> alignment_;
};

// The order refers to the number of previous alignments being remembered in the
// alignment lattice.
template <uint8_t order>
class MonotonicAligner {
 private:
  // The alignment lattice would normally be two dimensional, but if this is a
  // higher order model it needs additional dimensions to hold the history of
  // alignment operations.
  typedef boost::multi_array<double, order + 2> AlignmentLattice;
  // The index for a state in the alignment lattice is also multi-dimensional.
  typedef boost::array<boost::detail::multi_array::index, order + 2>
      AlignmentState;

 public:
  // The MonotonicAligner must be initialized with the maximum size of the
  // alignment chunks. This should be at least 2 (to allow 1:1 alignments).
  MonotonicAligner(uint8_t max_chunk_size);
  ~MonotonicAligner() {}

  // Returns the score of the observed alignment and calls ObservedArcUpdate on
  // each arc in the alignment.
  double ScoreObservedAlignment(SequencePair<order>* labeled_pair) const;
  // Runs the forward-backward algorithm on the given data, calling
  // GetScoreAndUpdate during the backward pass, and returns the pathsum.
  double ForwardBackward(SequencePair<order>* sequence_pair) const;

  // Find the highest scoring alignment for the given sequence pair, set its
  // internal alignment variable, and return the score of the best path.
  double Align(SequencePair<order>* sequence_pair) const;

 private:
  // Creates an alignment lattice for the given sequence pair.
  // The caller must delete the AlignmentLattice.
  AlignmentLattice* CreateAlignmentLattice(
      const SequencePair<order>& sequence_pair) const;
  // These two functions traverse an arc forwards or backwards to get the
  // resulting source/sink state. For GetSinkState, the align_op is the
  // alignment operation used to reach the sink state, and for GetSourceState,
  // align_op is the first alignment operation in the source's history.
  void GetSinkState(const AlignmentState& source,
                    const AlignmentOperation align_op,
                    AlignmentState* sink) const;
  void GetSourceState(const AlignmentState& sink,
                      const AlignmentOperation align_op,
                      AlignmentState* source) const;
  // Determines whether or not the given state is valid given the sequence pair.
  // The state may be invalid if it has an impossible history, or if its indices
  // don't fall within the sequence pair's boundaries.
  bool IsValid(const AlignmentState& state,
               const SequencePair<order>& sequence_pair) const;
  // Increment and Decrement state are used to navigate through the alignment
  // lattice, and return false if trying to increment past the last state or
  // decrement at the first state. If false is returned, the state's value will
  // be invalid.
  bool IncrementState(const SequencePair<order>& pair,
                      AlignmentState* state) const;
  bool DecrementState(const SequencePair<order>& pair,
                      AlignmentState* state) const;
  // This is guaranteed to be at least 2 by the constructor.
  const uint8_t max_chunk_size_;
  // Holds the number of alignment operations based on the chunk size.
  int max_alignment_op_;
};

#endif
