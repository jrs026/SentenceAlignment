#ifndef MONOTONIC_ALIGNER_H
#define MONOTONIC_ALIGNER_H

// MonotonicAligner.h
//
// A generic alignment model which allows m:n alignments and can incorporate
// some amount of alignment history. The SequencePair class allows different
// kinds of models (discriminative or generative).

#include <cstdlib>
#include <iostream>
#include <vector>

#include "boost/multi_array.hpp"
#include "boost/array.hpp"

using std::vector;

typedef uint16_t AlignmentOperation;

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

  // Sequence pairs are a nested class of MonotonicAligner so that they can have
  // the same order parameter.
  class SequencePair {
   public:
    SequencePair() {}
    virtual ~SequencePair() {}

    // These functions return the lengths of the sequences, and should not
    // change.
    inline int GetInputLength() { return input_length_; }
    inline int GetOutputLength() { return output_length_; }

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

   protected:
    // Returns the input/output spans given and alignment state and alignment
    // operation taken at that state.
    inline void SpanFromAlignmentArc(const AlignmentState& state,
        const AlignmentOperation align_op,
        int* i1, int* i2, int* o1, int* o2) const {
      *i1 = *i2 = state[0];
      *o1 = *o2 = state[1];
      MonotonicAligner<order>::UpdatePositions(align_op, false, i2, o2);
    }
    // These must be set by the subclass and left unchanged.
    int input_length_;
    int output_length_;

   private:
    vector<AlignmentOperation> alignment_;
  };

  // Returns the score of the observed alignment and calls ObservedArcUpdate on
  // each arc in the alignment.
  double ScoreObservedAlignment(SequencePair* labeled_pair) const;
  // Runs the forward-backward algorithm on the given data, calling
  // GetScoreAndUpdate during the backward pass, and returns the pathsum.
  double ForwardBackward(SequencePair* sequence_pair) const;

  // Find the highest scoring alignment for the given sequence pair, set its
  // internal alignment variable, and return the score of the best path.
  double Align(SequencePair* sequence_pair) const;

  // Update the positions based on the alignment operation taken. If reversed is
  // true, update the positions by going backwards instead of forwards.
  static inline void UpdatePositions(const AlignmentOperation align_op,
      bool reversed, int* input_pos, int* output_pos) {
    int update = (reversed ? -1 : 1);
    switch (align_op) {
      case 0: // 1:0 (Deletion)
        *input_pos += update;
        break;
      case 1: // 0:1 (Insertion)
        *output_pos += update;
        break;
      case 2: // 1:1 (Substitution)
        *input_pos += update;
        *output_pos += update;
        break;
      default:
        std::cerr << "Not yet implemented" << std::endl;
        exit(-1);
    }
  }

 private:
  // Creates an alignment lattice for the given sequence pair.
  // The caller must delete the AlignmentLattice.
  void CreateAlignmentLattice(const SequencePair& sequence_pair,
                              AlignmentLattice* lattice) const;
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
               const SequencePair& sequence_pair) const;
  // This is guaranteed to be at least 2 by the constructor.
  const uint8_t max_chunk_size_;
  // Holds the number of alignment operations based on the chunk size.
  int max_alignment_op_;
};

#endif
