// TODO: Move this into a test class

#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "alignment_models/edit_distance.h"
#include "alignment_models/model1.h"
#include "alignment_models/monotonic_aligner.h"
#include "util/math_util.h"
#include "util/parallel_corpus.h"
#include "util/vocab.h"

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::pair;
using std::vector;

void CreateRandomParams(EditDistanceParams* params) {
  double total = 0.0;
  for (int i = 0; i < params->alphabet_size(); ++i) {
    for (int j = 0; j < params->alphabet_size(); ++j) {
      params->at(i, j) = ((double)rand()/(double)RAND_MAX);
      total += params->at(i, j);
    }
  }
  params->at(0, 0) += params->alphabet_size();
  total += params->alphabet_size();
  for (int i = 0; i < params->alphabet_size(); ++i) {
    for (int j = 0; j < params->alphabet_size(); ++j) {
      params->at(i, j) = log(params->at(i, j) / total);
      cout << i << " " << j << " : " << params->at(i, j) << endl;
    }
  }
}

void SampleFromParams(const EditDistanceParams& params,
                      pair<int, int>* int_pair) {
  double rand_val = ((double)rand()/(double)RAND_MAX);
  double prob = 0.0;
  for (int i = 0; i < params.alphabet_size(); ++i) {
    for (int j = 0; j < params.alphabet_size(); ++j) {
      prob += exp(params.at(i, j));
      if (prob > rand_val) {
        int_pair->first = i;
        int_pair->second = j;
        return;
      }
    }
  }
  int_pair->first = params.alphabet_size() - 1;
  int_pair->second = params.alphabet_size() - 1;
}

void CreateRandomPair(const EditDistanceParams& params,
                      pair<string, string>* string_pair) {
  string alphabet = "abcdefghijklmnopqrstuvwxyz";
  bool done = false;
  while (!done) {
    pair<int, int> sample;
    SampleFromParams(params, &sample);
    if (sample.first > 0) {
      string_pair->first += alphabet[sample.first-1];
    }
    if (sample.second > 0) {
      string_pair->second += alphabet[sample.second-1];
    }
    if ((sample.first == 0) && (sample.second == 0)) {
      done = true;
    }
  }
  cout << string_pair->first << " " << string_pair->second << endl;
}

int main(int argc, char** argv) {
  MathUtil::InitLogTable();
  srand(time(NULL));
  MonotonicAligner<0> aligner1(2);
  MonotonicAligner<1> aligner2(2);
  Vocab vocab;
  ParallelCorpus pc(true);
  Model1 m1;

  bool edit_distance_test = false;
  if (edit_distance_test) {
    const int alphabet_size = 3; // Includes epsilon
    EditDistanceParams params(alphabet_size);
    CreateRandomParams(&params);
    EditDistanceModel<0> model(alphabet_size);
    vector<pair<string, string> > training_data;
    for (int i = 0; i < 100000; ++i) {
      pair<string, string> string_pair;
      CreateRandomPair(params, &string_pair);
      training_data.push_back(string_pair);
    }
    cerr << "Finished generating data" << endl;
    cout << model.EM(training_data, 50) << endl;
    string parameters;
    model.PrintParams(&parameters);
    cout << parameters;
  }
  return 0;
}
