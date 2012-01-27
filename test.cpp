#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>

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
using std::ifstream;
using std::make_pair;
using std::ofstream;
using std::pair;
using std::string;
using std::vector;

typedef ParallelCorpus::Sentence Sentence;

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

void ReadLinksFile(const string& links_file,
                   vector<string>* titles) {
  typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
  boost::char_separator<char> sep("\t");
  string line;

  ifstream in(links_file.c_str());
  if (!in) {
    cerr << "Error reading " << links_file << endl;
    exit(-1);
  }
  titles->clear();
  while (getline(in, line)) {
    tokenizer line_tokenizer(line, sep);
    int i = 0;
    for (tokenizer::iterator it = line_tokenizer.begin();
         it != line_tokenizer.end(); ++it) {
      if (i == 1) {
        titles->push_back(*it);
      }
      ++i;
    }
    assert(i == 2);
  }
  in.close();
}

// Used for sorting possible target sentences that align to a source sentence.
// (sorts in descending order)
bool CandidateSort(pair<int, double> a, pair<int, double> b) {
  return (a.second > b.second);
}

int main(int argc, char** argv) {
  MathUtil::InitLogTable();
  srand(time(NULL));
  ParallelCorpus pc(true);
  vector<string> source_files, target_files;
  source_files.push_back("data/dev_es_2008.tok");
  target_files.push_back("data/dev_en_2008.tok");
  source_files.push_back("data/dev_es_2009.tok");
  target_files.push_back("data/dev_en_2009.tok");
  source_files.push_back("data/dev_es_2010.tok");
  target_files.push_back("data/dev_en_2010.tok");
  //source_files.push_back("data/europarl_10k_es.tok");
  //target_files.push_back("data/europarl_10k_en.tok");
  //source_files.push_back("data/europarl-v6.es-en.es.tok");
  //target_files.push_back("data/europarl-v6.es-en.en.tok");

  // Determine which tests are run.
  bool naacl_wiki_test = false;
  bool build_m1_filter = false;
  bool filter_sentences_test = false;
  bool create_turk_csvs = true;
  bool document_aligner_test = false;
  bool model1_test = false;
  bool edit_distance_test = false;

  // Parameters for the tests
  double m1_prior = 1.1;
  bool stemming = false;
  vector<string> langs;
  //langs.push_back("bn");
  //langs.push_back("hi");
  //langs.push_back("ml");
  //langs.push_back("ta");
  //langs.push_back("te");
  langs.push_back("ur");

  // For filter_sentences_test and create_turk_csvs
  double cutoff = 0.0001;
  double log_word_cutoff = log(0.05);
  bool covered_unk = true;
  bool ignored_unk = false;
  int min_candidates = 5; // Minimum # of candidate targets
  int candidates_per_task = 10;

  cout.precision(4);
  if (naacl_wiki_test) {
    int m1_iterations = 2;
    int doc_iterations = 2;
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
            "data/esen_docs_small.source",
            "data/esen_docs_small.target")) {
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
    DocumentAligner<0> aligner(&pc, 0.2, m1_prior, poisson_lm);
    for (int i = 0; i < m1_iterations; ++i) {
      cout << "Parallel Sentence EM Iteration " << i + 1 << endl;
      Model1* m1 = aligner.MutableModel1();
      for (int j = doc_max; j < pc.size(); ++j) {
        if ((pc.GetDocPair(j).first.size() != 1) 
          || (pc.GetDocPair(j).second.size() != 1)) {
          cout << "Document " << j << " missing sentences" << endl;
        }
        //pc.PrintSentencePair(pc.GetDocPair(j).first.at(0),
        //    pc.GetDocPair(j).second.at(0), cout);
        //cout << endl;
        m1->EStep(pc.GetDocPair(j).first.at(0),
                  pc.GetDocPair(j).second.at(0), 0.0);
      }
      m1->MStep(is_variational);
    }
    cout << "Finished parallel sentence EM" << endl;
    //aligner.GetModel1().PrintTTable(
    //    pc.source_vocab(), pc.target_vocab(), cout);
    for (double lambda = 1e-12; lambda >= 1e-12; lambda /= 10) {
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

  if (build_m1_filter) {
    for (int l = 0; l < langs.size(); ++l) {
      string base = "/home/hltcoe/jsmith/wiki/indian-data/";
      string source_file = base + langs[l] + "-en/training_dict." + langs[l]
          + "-en." + langs[l];
      string target_file = base + langs[l] + "-en/training_dict." + langs[l]
          + "-en.en";
      string st_out_file = base + langs[l] + ".st_dict.bin";
      string st_out_svocab = base + langs[l] + ".st_dict.svocab";
      string st_out_tvocab = base + langs[l] + ".st_dict.tvocab";
      string ts_out_file = base + langs[l] + ".ts_dict.bin";
      string ts_out_svocab = base + langs[l] + ".ts_dict.svocab";
      string ts_out_tvocab = base + langs[l] + ".ts_dict.tvocab";
      ParallelCorpus st_pc(true);
      ParallelCorpus ts_pc(true);
      if (stemming) {
        st_pc.SetSourceStemming(true);
        ts_pc.SetTargetStemming(true);
      }
      if (!st_pc.ReadParallelData(source_file, target_file)) {
        cerr << "Error reading document pair" << endl;
        exit(-1);
      }
      if (!ts_pc.ReadParallelData(target_file, source_file)) {
        cerr << "Error reading document pair" << endl;
        exit(-1);
      }
      Model1 st_m1(m1_prior);
      Model1 ts_m1(m1_prior);
      vector<const ParallelCorpus*> st_pcs, ts_pcs;
      // s->t
      st_pcs.push_back(&st_pc);
      st_m1.InitDataStructures(st_pcs, st_pc.source_vocab(), st_pc.target_vocab());
      st_m1.ClearExpectedCounts();
      for (int i = 0; i < 10; ++i) {
        double likelihood = 0.0;
        for (int j = 0; j < st_pc.size(); ++j) {
          if ((st_pc.GetDocPair(j).first.size() != 1) 
            || (st_pc.GetDocPair(j).second.size() != 1)) {
            cout << "Document " << j << " missing sentences" << endl;
          }
          likelihood += st_m1.EStep(st_pc.GetDocPair(j).first.at(0),
                                 st_pc.GetDocPair(j).second.at(0), 0.0);
        }
        cout << "Iteration " << i + 1 << " likelihood: " << likelihood << endl;
        st_m1.MStep(false);
        st_m1.ClearExpectedCounts();
      }
      // t->s
      ts_pcs.push_back(&ts_pc);
      ts_m1.InitDataStructures(ts_pcs, ts_pc.source_vocab(), ts_pc.target_vocab());
      ts_m1.ClearExpectedCounts();
      for (int i = 0; i < 10; ++i) {
        double likelihood = 0.0;
        for (int j = 0; j < ts_pc.size(); ++j) {
          if ((ts_pc.GetDocPair(j).first.size() != 1) 
            || (ts_pc.GetDocPair(j).second.size() != 1)) {
            cout << "Document " << j << " missing sentences" << endl;
          }
          likelihood += ts_m1.EStep(ts_pc.GetDocPair(j).first.at(0),
                                 ts_pc.GetDocPair(j).second.at(0), 0.0);
        }
        cout << "Iteration " << i + 1 << " likelihood: " << likelihood << endl;
        ts_m1.MStep(false);
        ts_m1.ClearExpectedCounts();
      }
      st_m1.WriteBinary(st_out_file, st_out_svocab, st_out_tvocab,
          st_pc.source_vocab(), st_pc.target_vocab());
      ts_m1.WriteBinary(ts_out_file, ts_out_svocab, ts_out_tvocab,
          ts_pc.source_vocab(), ts_pc.target_vocab());
      cout << "Finished " << langs[l] << "-en" << endl;
    }
  }

  if (filter_sentences_test) {
    for (int l = 0; l < langs.size(); ++l) {
      string base = "/home/hltcoe/jsmith/wiki/indian-data/";
      string st_m1_file = base + langs[l] + ".st_dict.bin";
      string st_svocab_file = base + langs[l] + ".st_dict.svocab";
      string st_tvocab_file = base + langs[l] + ".st_dict.tvocab";
      string ts_m1_file = base + langs[l] + ".ts_dict.bin";
      string ts_svocab_file = base + langs[l] + ".ts_dict.svocab";
      string ts_tvocab_file = base + langs[l] + ".ts_dict.tvocab";
      Vocab st_source_vocab, st_target_vocab;
      Model1 st_m1(m1_prior);
      st_m1.InitFromBinaryFile(st_m1_file, st_svocab_file, st_tvocab_file,
          &st_source_vocab, &st_target_vocab);
      Vocab ts_source_vocab, ts_target_vocab;
      Model1 ts_m1(m1_prior);
      ts_m1.InitFromBinaryFile(ts_m1_file, ts_svocab_file, ts_tvocab_file,
          &ts_source_vocab, &ts_target_vocab);
      cout << "Finished reading Model 1 files" << endl;
      // Init data structures from the written dictionaries
      string source_file = base + langs[l] + "-en/devtest." + langs[l]
          + "-en." + langs[l];
      int total_pos = 0;
      int total_neg = 0;
      double true_pos = 0;
      double false_pos = 0;
      vector<string> score_types;
      score_types.push_back("st_m1");
      score_types.push_back("ts_m1");
      score_types.push_back("st_m1_viterbi");
      score_types.push_back("ts_m1_viterbi");
      score_types.push_back("st_coverage");
      score_types.push_back("ts_coverage");
      score_types.push_back("coverage");
      vector<double> pos_scores(score_types.size(), 0.0);
      vector<double> neg_scores(score_types.size(), 0.0);
      // Iterate over the different references.
      for (int f = 0; f < 1; ++f) {
        string target_file = base + langs[l] + "-en/devtest." + langs[l]
            + "-en.en." + util::ToString(f) + ".br";
        ParallelCorpus st_pc(true);
        ParallelCorpus ts_pc(true);
        if (stemming) {
          st_pc.SetSourceStemming(true);
          ts_pc.SetTargetStemming(true);
        }
        st_pc.AddSourceVocab(st_source_vocab);
        st_pc.AddTargetVocab(st_target_vocab);
        if (!st_pc.ReadParallelData(source_file, target_file)) {
          cerr << "Error reading document pair" << endl;
          exit(-1);
        }
        ts_pc.AddSourceVocab(ts_source_vocab);
        ts_pc.AddTargetVocab(ts_target_vocab);
        if (!ts_pc.ReadParallelData(target_file, source_file)) {
          cerr << "Error reading document pair" << endl;
          exit(-1);
        }
        for (int i = 0; i < st_pc.size(); ++i) {
          const Sentence& st_source = st_pc.GetDocPair(i).first.at(0);
          const Sentence& ts_target = ts_pc.GetDocPair(i).second.at(0);
          int source_size = st_source.size();
          for (int j = 0; j < st_pc.size(); ++j) {
          //for (int j = i - 5; j < i + 5; ++j) {
            if ((j < 0) || (j >= st_pc.size())) {
              continue;
            }
            const Sentence& st_target = st_pc.GetDocPair(j).second.at(0);
            const Sentence& ts_source = ts_pc.GetDocPair(j).first.at(0);
            int target_size = st_target.size();
            double st_m1_score = math_util::Poisson(source_size, target_size)
                * exp(st_m1.ScorePair(st_source, st_target) / target_size);
            double ts_m1_score = math_util::Poisson(target_size, source_size)
                * exp(ts_m1.ScorePair(ts_source, ts_target) / source_size);
                /*
            double st_m1_score_v = math_util::Poisson(source_size, target_size)
                * exp(st_m1.ViterbiScorePair(st_source, st_target) / target_size);
            double ts_m1_score_v = math_util::Poisson(target_size, source_size)
                * exp(ts_m1.ViterbiScorePair(ts_source, ts_target) / source_size);
                */
            /*
            double st_m1_score = 0.0;
            double ts_m1_score = 0.0;
            */
            double st_m1_score_v = 0.0;
            double ts_m1_score_v = 0.0;
            double st_cov = st_m1.ComputeCoverage(st_pc.GetDocPair(i).first.at(0),
                st_pc.GetDocPair(j).second.at(0), log_word_cutoff, covered_unk,
                ignored_unk);
            double ts_cov = ts_m1.ComputeCoverage(ts_pc.GetDocPair(j).first.at(0),
                ts_pc.GetDocPair(i).second.at(0), log_word_cutoff, covered_unk,
                ignored_unk);
            if (i == j) {
              ++total_pos;
              pos_scores[0] += st_m1_score;
              pos_scores[1] += ts_m1_score;
              pos_scores[2] += st_m1_score_v;
              pos_scores[3] += ts_m1_score_v;
              pos_scores[4] += st_cov;
              pos_scores[5] += ts_cov;
              pos_scores[6] += st_cov + ts_cov;
              //if (st_cov + ts_cov >= cutoff) {
              if (((st_m1_score + ts_m1_score) / 2) >= cutoff) {
                ++true_pos;
              }
            } else {
              ++total_neg;
              neg_scores[0] += st_m1_score;
              neg_scores[1] += ts_m1_score;
              neg_scores[2] += st_m1_score_v;
              neg_scores[3] += ts_m1_score_v;
              neg_scores[4] += st_cov;
              neg_scores[5] += ts_cov;
              neg_scores[6] += st_cov + ts_cov;
              //if (st_cov + ts_cov >= cutoff) {
              if (((st_m1_score + ts_m1_score) / 2) >= cutoff) {
                ++false_pos;
              }
            }
          }
        }
      }
      cout << langs[l] << ":" << endl;
      for (int sc = 0; sc < score_types.size(); ++sc) {
        cout << "Average positive " << score_types[sc] << " score: "
            << (pos_scores[sc] / total_pos) << endl;
        cout << "Average negative " << score_types[sc] << " score: "
            << (neg_scores[sc] / total_neg) << endl;
      }
      cout << "Recall: " << (true_pos / total_pos) * 100 << "%" << endl;
      cout << "Precision: " << (true_pos / (false_pos + true_pos)) * 100
          << "%" << endl;
      cout << "Pruning rate: " << (1.0 - (false_pos / total_neg)) * 100 << "%" << endl;
      cout << endl;
    }
  }

  if (create_turk_csvs) {
    for (int l = 0; l < langs.size(); ++l) {
      string base = "/home/hltcoe/jsmith/wiki/indian-data/";
      string st_m1_file = base + langs[l] + ".st_dict.bin";
      string st_svocab_file = base + langs[l] + ".st_dict.svocab";
      string st_tvocab_file = base + langs[l] + ".st_dict.tvocab";
      string ts_m1_file = base + langs[l] + ".ts_dict.bin";
      string ts_svocab_file = base + langs[l] + ".ts_dict.svocab";
      string ts_tvocab_file = base + langs[l] + ".ts_dict.tvocab";
      Vocab st_source_vocab, st_target_vocab;
      Model1 st_m1(m1_prior);
      st_m1.InitFromBinaryFile(st_m1_file, st_svocab_file, st_tvocab_file,
          &st_source_vocab, &st_target_vocab);
      Vocab ts_source_vocab, ts_target_vocab;
      Model1 ts_m1(m1_prior);
      ts_m1.InitFromBinaryFile(ts_m1_file, ts_svocab_file, ts_tvocab_file,
          &ts_source_vocab, &ts_target_vocab);
      cout << "Finished reading Model 1 files" << endl;
      
      string wiki_base = "/home/hltcoe/jsmith/wiki/data/";
      string links_file = wiki_base + langs[l] + "/" + langs[l] + "-en100.links";
      vector<string> titles;
      ReadLinksFile(links_file, &titles);
      int total_pos = 0;
      int total_neg = 0;
      int total_tasks = 0;
      int total_source_sents = 0;
      int total_target_sents = 0;
      // Iterate over the different titles:
      for (int t = 0; t < titles.size(); ++t) {
        string title = titles[t];
        cout << title << ":" << endl;
        string pair_file = 
            wiki_base + langs[l] + "/" + title + ".pruned_pairs";
        ofstream out(pair_file.c_str());
        int pos = 0;
        int neg = 0;
        int tasks = 0;
        int source_sents = 0;
        int target_sents = 0;
        string source_file = wiki_base + langs[l] + "/" + title + ".source.br";
        string target_file = wiki_base + "en/" + title + ".target.br";
        ParallelCorpus st_pc(true);
        ParallelCorpus ts_pc(true);
        if (stemming) {
          st_pc.SetSourceStemming(true);
          ts_pc.SetTargetStemming(true);
        }
        st_pc.AddSourceVocab(st_source_vocab);
        st_pc.AddTargetVocab(st_target_vocab);
        if (!st_pc.ReadDocumentPairs(source_file, target_file)) {
          cerr << "Skipping document pair " << title << endl;
          continue;
        }
        ts_pc.AddSourceVocab(ts_source_vocab);
        ts_pc.AddTargetVocab(ts_target_vocab);
        if (!ts_pc.ReadDocumentPairs(target_file, source_file)) {
          cerr << "Skipping document pair " << title << endl;
          continue;
        }
        for (int i = 0; i < st_pc.GetDocPair(0).first.size(); ++i) {
          vector<pair<int, double> > candidates;
          const Sentence& st_source = st_pc.GetDocPair(0).first.at(i);
          const Sentence& ts_target = ts_pc.GetDocPair(0).second.at(i);
          int source_size = st_source.size();
          for (int j = 0; j < st_pc.GetDocPair(0).second.size(); ++j) {
            const Sentence& st_target = st_pc.GetDocPair(0).second.at(j);
            const Sentence& ts_source = ts_pc.GetDocPair(0).first.at(j);
            int target_size = st_target.size();
            double st_m1_score = math_util::Poisson(source_size, target_size)
                * exp(st_m1.ScorePair(st_source, st_target) / target_size);
            double ts_m1_score = math_util::Poisson(target_size, source_size)
                * exp(ts_m1.ScorePair(ts_source, ts_target) / source_size);

            candidates.push_back(make_pair(j, (double)
                                           (st_m1_score + ts_m1_score) / 2));
          }
          // sort
          sort(candidates.begin(), candidates.end(), CandidateSort);
          // Always keep the top n
          for (int c = min_candidates; c < candidates.size(); ++c) {
            if (candidates[c].second < cutoff) {
              candidates.resize(c);
            }
          }
          pos += candidates.size();
          neg += st_pc.GetDocPair(0).second.size() - candidates.size();

          // Shuffle and output the candidates into tasks
          std::random_shuffle(candidates.begin(), candidates.end());
          while (candidates.size() > 0) {
            int num_candidates =
                std::min<int>(candidates.size(), candidates_per_task);
            out << title << endl;
            out << num_candidates << endl;
            out << st_pc.source_vocab().ToText(st_source) << endl;
            out << i << endl;
            while (num_candidates > 0) {
              out << st_pc.target_vocab().ToText(
                  st_pc.GetDocPair(0).second.at(candidates.back().first)) << endl;
              out << candidates.back().first << endl;
              candidates.pop_back();
              num_candidates--;
            }
            ++tasks;
          }
        }
        source_sents = st_pc.GetDocPair(0).first.size();
        target_sents = st_pc.GetDocPair(0).second.size();
        total_pos += pos;
        total_neg += neg;
        total_tasks += tasks;
        total_source_sents += source_sents;
        total_target_sents += target_sents;
        cout << "Positive pairs: " << pos << endl;
        cout << "All possible pairs: " << pos + neg << " ("
            << 100.0 - (((double) pos / (pos + neg)) * 100) << "% pruning)" << endl;
        cout << "Tasks: " << tasks << endl;
        cout << "Source sentences: " << source_sents << endl;
        cout << "Target sentences: " << target_sents << endl;
        cout << endl;
        out.close();
      }
      cout << langs[l] << " total:" << endl;
      cout << "Positive pairs: " << total_pos << endl;
      cout << "All possible pairs: " << total_pos + total_neg << " ("
          << 100.0 - (((double) total_pos / (total_pos + total_neg)) * 100)
          << "% pruning)" << endl;
      cout << "Total Tasks: " << total_tasks << endl;
      cout << "Total source sentences: " << total_source_sents << endl;
      cout << "Total target sentences: " << total_target_sents << endl;
      cout << endl;
    }
  }

  if (document_aligner_test) {
    int iterations = 100;
    for (double del_percent = 0.1; del_percent <= 0.31; del_percent += 0.1) {
      srand(1);
      cout << "With " << del_percent * 100 << "% of sentences deleted:" << endl;
      pc.ClearData();
      for (int i = 0; i < source_files.size(); ++i) {
        if (!pc.ReadAlignedPairs(source_files.at(i), target_files.at(i))) {
          cerr << "Error reading document pair: (" << source_files.at(i) << ", "
               << target_files.at(i) << ")" << endl;
        }
      }
      cout << "Using " << pc.size() << " documents:" << endl;
      pc.RandomDeletion(del_percent);
      pc.PrintStats(cout);
      double b_precision, b_recall, b_f1;
      pc.DiagonalBaseline(&b_precision, &b_recall, &b_f1);
      cout << "Baseline:"
           << "\tPrecision: " << b_precision * 100 
           << "\tRecall: " << b_recall * 100
           << "\tF1: " << b_f1 * 100 << endl;
      for (double lambda = 3 - (2 * sqrt(2)); lambda < 1.0; lambda += 1.0) {
      //for (double lambda = 0.05; lambda < 1.0; lambda += 0.05) {
        cout << endl << "Lambda = " << lambda << endl;
        DocumentAligner<0> aligner(&pc, lambda, 1.01, true);
        for (int i = 0; i < iterations; ++i) {
          cout << aligner.EM(false) << endl;
          double precision, recall, f1;
          aligner.Test(pc.size(), &precision, &recall, &f1);
          cout << "Iteration " << i + 1
               << ":\tPrecision: " << precision * 100 
               << "\tRecall: " << recall * 100
               << "\tF1: " << f1 * 100 << endl;
        }
        //aligner.GetModel1().PrintTTable(
        //    pc.source_vocab(), pc.target_vocab(), cout);
      }
      cout << endl;
    }
  }

  if (model1_test) {
    if (pc.ReadAlignedPairs("data/source.txt", "data/target.txt")) {
      const ParallelCorpus::DocumentPair& doc_pair = pc.GetDocPair(0);
      Model1 m1;
      vector<const ParallelCorpus*> pcs;
      pcs.push_back(&pc);
      m1.InitDataStructures(pcs, pc.source_vocab(), pc.target_vocab());
      m1.ClearExpectedCounts();
      for (int i = 0; i < 100; ++i) {
        double likelihood = 0.0;
        for (int s = 0; s < doc_pair.first.size(); ++s) {
          for (int t = 0; t < doc_pair.second.size(); ++t) {
            const ParallelCorpus::Sentence& source = doc_pair.first.at(s);
            const ParallelCorpus::Sentence& target = doc_pair.second.at(t);
            double weight = log(1.0 / (1.0 + (1 * abs(s - t))));
            likelihood += m1.EStep(source, target, weight) * exp(weight);
          }
        }
        cout << likelihood << endl;
        m1.MStep(false);
        m1.ClearExpectedCounts();
      }
    } else {
      cerr << "Error reading document pair" << endl;
    }
  }

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
