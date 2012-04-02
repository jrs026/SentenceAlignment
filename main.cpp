#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <boost/program_options.hpp>

#include "alignment_models/document_aligner.h"
#include "alignment_models/edit_distance.h"
#include "alignment_models/model1.h"
#include "alignment_models/monotonic_aligner.h"
#include "util/math_util.h"
#include "util/parallel_corpus.h"
#include "util/util.h"
#include "util/vocab.h"

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::pair;
using std::vector;

namespace po = boost::program_options;

int main(int argc, char** argv) {
  MathUtil::InitLogTable();
  srand(time(NULL));
  
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help", "produce help message")
  ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);    

  if (vm.count("help")) {
      cout << desc << "\n";
      return 1;
  }

  ParallelCorpus pc(true);
  vector<string> source_files, target_files;
  source_files.push_back("data/dev_es_2008.tok");
  target_files.push_back("data/dev_en_2008.tok");
  source_files.push_back("data/dev_es_2009.tok");
  target_files.push_back("data/dev_en_2009.tok");
  source_files.push_back("data/dev_es_2010.tok");
  target_files.push_back("data/dev_en_2010.tok");
  source_files.push_back("data/europarl_10k_es.tok");
  target_files.push_back("data/europarl_10k_en.tok");
  //source_files.push_back("data/europarl-v6.es-en.es.tok");
  //target_files.push_back("data/europarl-v6.es-en.en.tok");

  bool naacl_wiki_test = true;
  if (naacl_wiki_test) {
    int m1_iterations = 0;
    int doc_iterations = 5;
    double m1_smoothing = 1.0;
    bool is_variational = false;
    bool poisson_lm = true;

    if (!pc.ReadAlignedPairs(
            "data/es.source.dev",
            "data/es.target.dev",
            "data/es.alignment.dev")) {
          //  "data/eswiki_q1.source.dev",
          //  "data/eswiki_q1.target.dev",
          //  "data/eswiki_q1.alignment.dev")) {
      cerr << "Error reading wiki documents." << endl;
      exit(-1);
    }
    int labeled_max = pc.size();
    if (!pc.ReadDocumentPairs(
            "data/esen_docs.source",
            "data/esen_docs.target")) {
      cerr << "Error reading wiki documents." << endl;
      exit(-1);
    }
    int doc_max = pc.size();
    for (int i = 0; i < source_files.size(); ++i) {
      if (!pc.ReadParallelData(source_files.at(i), target_files.at(i))) {
        cerr << "Error reading document pair: (" << source_files.at(i) << ", "
             << target_files.at(i) << ")" << endl;
      }
    }

    cout << "Using " << pc.size() << " documents:" << endl;
    DocumentAligner<0> aligner(&pc, 0.2, m1_smoothing, poisson_lm);
    for (int i = 0; i < m1_iterations; ++i) {
      cout << "Parallel Sentence EM Iteration " << i + 1 << endl;
      Model1* m1 = aligner.MutableModel1();
      for (int j = doc_max; j < pc.size(); ++j) {
        if ((pc.GetDocPair(j).first.size() != 1) 
          || (pc.GetDocPair(j).second.size() != 1)) {
          cout << "Document " << j << " missing sentences" << endl;
        }
        m1->EStep(pc.GetDocPair(j).first.at(0),
                  pc.GetDocPair(j).second.at(0), 0.0);
      }
      m1->MStep(is_variational);
    }
    cout << "Finished parallel sentence EM" << endl;
    //aligner.GetModel1().PrintTTable(pc, cout);
    for (double lambda = 1e-10; lambda >= 1e-50; lambda /= 10) {
    //for (double lambda = 3 - (2 * sqrt(2)); lambda < 1.0; lambda += 1.0) {
      cout << endl << "Lambda = " << lambda << endl;
      aligner.SetLambda(lambda);
      double precision, recall, f1;
      aligner.Test(labeled_max, &precision, &recall, &f1);
      cout << "Iteration 0"
           << ":\tPrecision: " << precision * 100 
           << "\tRecall: " << recall * 100
           << "\tF1: " << f1 * 100 << endl;
      for (int i = 0; i < doc_iterations; ++i) {
        cout << aligner.EM(is_variational, doc_max) << endl;
        aligner.Test(labeled_max, &precision, &recall, &f1);
        cout << "Iteration " << i + 1
             << ":\tPrecision: " << precision * 100 
             << "\tRecall: " << recall * 100
             << "\tF1: " << f1 * 100 << endl;
      }
    }
    cout << endl;
  }

  return 0;
}
