#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <boost/filesystem.hpp>
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

namespace fs = boost::filesystem;
namespace po = boost::program_options;

int main(int argc, char** argv) {
  MathUtil::InitLogTable();
  srand(time(NULL));

  vector<string> parallel_data, comparable_data;
  string model_type = "m1";
  int wa_iterations = 5;
  int doc_iterations = 5;
  string load_model_file = "";
  string save_model_file = "";
  string parallel_output_location = "";
  bool use_lowercase = true;
  double emission_smoothing = 1.0;
  double transition_smoothing = 1.0;
  int distortion_window = 5;
  double null_transition_prob = 0.2;
  bool use_poisson_lm = true;
  double doc_alignment_prior = 1e-12;
  bool use_variational = false;
  
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help", "produce help message")
      ("parallel_data,p", po::value<vector<string> >(&parallel_data), 
        "Sentence aligned data")
      ("comparable_data,c", po::value<vector<string> >(&comparable_data),
        "Document aligned comparable data")
      ("wa_model", po::value<string>(&model_type),
        "The type of word alignment model to be used ('*m1' or 'hmm')")
      ("wa_iterations", po::value<int>(&wa_iterations),
        "Number of EM iterations for word alignment on parallel data (default is 5)")
      ("doc_iterations", po::value<int>(&doc_iterations),
        "Number of EM iterations for the comparable document alignment model (default is 5)")
      ("load_model,l", po::value<string>(&load_model_file),
        "Loads word alignment model parameters")
      ("save_model,s", po::value<string>(&save_model_file),
        "Saves word alignment model parameters")
      ("parallel_output", po::value<string>(&parallel_output_location),
        "Prints parallel sentences extracted from the comparable data to this location")
      ("lowercase", po::value<bool>(&use_lowercase),
        "Lowercase the parallel/comparable data")
      ("emission_smoothing", po::value<double>(&emission_smoothing),
        "Smoothing for the p(t|s) parameters of the word alignment model")
      ("transition_smoothing", po::value<double>(&transition_smoothing),
        "Smoothing for the distortion parameters of the HMM word alignment model")
      ("distortion_window", po::value<int>(&distortion_window),
        "Size of the distortion window in the HMM (default is 5)")
      ("null_transition_prob", po::value<double>(&null_transition_prob),
        "Probability of transitioning to the null state in the HMM (default is 0.2)")
      ("poission_lm", po::value<bool>(&use_poisson_lm),
        "Use a Poisson distribution for modeling length in the document aligner's"
        " language model (default is true)")
      ("doc_alignment_prior", po::value<double>(&doc_alignment_prior),
        "Mixture weight for the bilingual/monolingual generation probabilities in"
        " the document aligner (default is 1e-12)")
      ("variational", po::value<bool>(&use_variational),
        "Use variational inference when updating word alignment model parameters"
        " (default is false)")
  ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);    

  if (vm.count("help")) {
      cout << desc << "\n";
      // TODO: Description of the expected suffixes
      return 1;
  }

  Model1* wa_model;
  if (model_type == "m1") {
    wa_model = new Model1();
  } else if (model_type == "hmm") {
    // TODO: Create generic word aligner class that HMM inherits from
    //wa_model = new HMMAligner();
    cout << "Not yet implemented" << endl;
    return 1;
  } else {
    cout << "Unknown word alignment model type: " << model_type << endl;
    return 1;
  }

  ParallelCorpus pc(use_lowercase);

  // If we are loading a saved model, load its vocabulary first and use the same
  // vocabulary for the parallel corpus object.
  if (load_model_file != "") {
    Vocab s_vocab;
    if (s_vocab.Read(load_model_file + ".svocab")) {
      cout << "Reading vocab from " << load_model_file << ".svocab" << endl;
      pc.AddSourceVocab(s_vocab);
    }
    Vocab t_vocab;
    if (t_vocab.Read(load_model_file + ".tvocab")) {
      cout << "Reading vocab from " << load_model_file << ".tvocab" << endl;
      pc.AddTargetVocab(t_vocab);
    }
  }

  // Read comparable data
  for (int i = 0; i < comparable_data.size(); ++i) {
    string source_file = comparable_data[i] + ".source";
    string target_file = comparable_data[i] + ".target";
    string alignment_file = comparable_data[i] + ".alignment";
    if (fs::exists(fs::path(alignment_file))) {
      if (!pc.ReadPartiallyAlignedPairs(source_file, target_file, alignment_file)) {
        cerr << "Error reading comparable docs: (" << source_file << ", "
             << target_file << ")" << endl;
      }
    } else {
      if (!pc.ReadDocumentPairs(source_file, target_file)) {
        cerr << "Error reading comparable docs: (" << source_file << ", "
             << target_file << ")" << endl;
      }
    }
  }
  int doc_max = pc.size();

  // Read parallel data
  for (int i = 0; i < parallel_data.size(); ++i) {
    string source_file = parallel_data[i] + ".source";
    string target_file = parallel_data[i] + ".target";
    if (!pc.ReadParallelData(source_file, target_file)) {
      cerr << "Error reading parallel data: (" << source_file << ", "
           << target_file << ")" << endl;
    }
  }
  // Both parallel data and comparable data is stored in the same parallel
  // corpus object, so we need to keep track of where the last bit of parallel
  // data is.
  int parallel_max = pc.size();
  cout << "Read " << parallel_max - doc_max << " sentence pairs." << endl;

  // Read a saved model if one is provided, otherwise initialize parameters from
  // the parallel data.
  // TODO: Handle the case where we read a model and need to add p(t|s)
  // parameters for unseen words in the parallel corpus we just read.
  if (load_model_file != "") {
    // TODO: Handle vocabularies correctly
    string s_vocab_file = load_model_file + ".svocab";
    string t_vocab_file = load_model_file + ".tvocab";
    //wa_model->InitFromBinaryFile(load_model_file, s_vocab_file, t_vocab_file);
  } else {
    vector<const ParallelCorpus*> pcs(1, &pc);
    wa_model->InitDataStructures(pcs, pc.source_vocab(), pc.target_vocab());
  }

  // Initialize the aligner (TODO: handle different aligners)
  DocumentAligner<0> aligner(
      &pc, doc_alignment_prior, emission_smoothing, use_poisson_lm);
  for (int i = 0; i < wa_iterations; ++i) {
    cout << "Parallel Sentence EM Iteration " << i + 1 << endl;
    Model1* m1 = aligner.MutableModel1();
    for (int j = doc_max; j < parallel_max; ++j) {
      if ((pc.GetDocPair(j).first.size() != 1) 
        || (pc.GetDocPair(j).second.size() != 1)) {
        cout << "Document " << j << " missing sentences" << endl;
      }
      m1->EStep(pc.GetDocPair(j).first.at(0),
                pc.GetDocPair(j).second.at(0), 0.0);
    }
    m1->MStep(use_variational);
  }
  cout << "Finished parallel sentence EM" << endl;
  double precision, recall, f1;
  aligner.Test(doc_max, &precision, &recall, &f1);
  cout << "Iteration 0"
       << ":\tPrecision: " << precision * 100 
       << "\tRecall: " << recall * 100
       << "\tF1: " << f1 * 100 << endl;
  for (int i = 0; i < doc_iterations; ++i) {
    cout << aligner.EM(use_variational, pc.size()) << endl;
    aligner.Test(doc_max, &precision, &recall, &f1);
    cout << "Iteration " << i + 1
         << ":\tPrecision: " << precision * 100 
         << "\tRecall: " << recall * 100
         << "\tF1: " << f1 * 100 << endl;
  }

  if (parallel_output_location != "") {
    aligner.ExtractSentences(doc_max, parallel_output_location);
  }

  delete wa_model;
  return 0;
}
