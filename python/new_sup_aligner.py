#!/usr/bin/python

import copy
import math
import os
import re
import sys

from optparse import OptionParser

import maxent
import sentence_pair_extractor

def main():
  parser = OptionParser()

  parser.add_option("-p", "--parallel_data", dest="training_file",
      default="data/euro_esen_10k",
      help="Parallel data, expecting \".source\" and \".target\"")

  parser.add_option("-c", "--comparable_data", dest="comp_data",
      default="data/es_dev",
      help="Annotated comparable data, expecting \".source\", \".target\", and \".alignment\"")

  parser.add_option("-r", "--raw-data", dest="raw_data",
      default="data/esen_docs_small",
      help="Raw comparable data, expecting \".source\" and \".target\"")

  parser.add_option("-t", "--t-table", dest="m1_data", default="small.model",
      help="Word alignment parameters from some parallel data")

  parser.add_option("-d", "--dictionary", dest="dictionary",
      default="data/esen.dict", help="Location of bilingual dictionary")

  parser.add_option("-e", "--example-window", dest="example_window", type="int",
      default=3, help="Size of the example window for gathering training data")

  parser.add_option("--length-ratio", type="float", dest="max_len_ratio",
      default=3.0, 
      help="Maximum length ratio for sentences to be considered parallel")

  parser.add_option("--test-max", type="int", dest="test_max", default=100,
      help="Number of sentences from the parallel data to use as test data")

  parser.add_option("--prob-floor", type="float", dest="prob_floor",
      default=1e-4, help="Lowest probability value for LM and M1")

  parser.add_option("--max_iterations", type="int", dest="max_iterations",
      default=20, help="Maximum number of L-BFGS iterations")

  parser.add_option("--l2-norm", type="float", dest="l2_norm", default="2.0",
      help="L2 normalizing value for the Maxent model")

  (opts, args) = parser.parse_args()
  spe = sentence_pair_extractor.SentencePairExtractor(opts)

  # Read available data
  if opts.training_file:
    (source_parallel, target_parallel) = spe.read_parallel_data(
        opts.training_file + '.source',
        opts.training_file + '.target')
    spe.create_lm(target_parallel)
  if opts.comp_data:
    (source_docs, target_docs, alignments) = spe.read_comp_data(opts.comp_data)
  if opts.raw_data:
    (raw_source, raw_target) = spe.read_comp_data(opts.raw_data)
  if opts.m1_data:
    #spe.read_m1_probs(opts.m1_data)
    spe.dict_from_m1(opts.m1_data)
  if opts.dictionary:
    spe.read_dictionary(opts.dictionary)

  spe.init_feature_functions()

  if opts.training_file:
    print "Performance on parallel data:"
    (parallel_train, parallel_test) = spe.create_annotated_parallel_data(
        source_parallel, target_parallel)
    print_feature_stats(parallel_train)

    spe.train_model(parallel_train)
    print spe.test_model(parallel_test)

  if opts.comp_data:
    print "Performance on comparable data:"
    annotated_comp_data = spe.create_annotated_comp_data(
        source_docs, target_docs, alignments)
    print_feature_stats(annotated_comp_data)

    # TODO: Testing on training data temporarily
    spe.train_model(annotated_comp_data)
    print spe.test_model(annotated_comp_data)

def print_feature_stats(train_data):
  count_by_outcome = {}
  feature_stats = {}
  feature_names = {}
  for (context, outcome) in train_data:
    if not count_by_outcome.get(outcome):
      count_by_outcome[outcome] = 0
    count_by_outcome[outcome] += 1
    for (name, value) in context:
      if not feature_names.get(name):
        feature_names[name] = True
      if not feature_stats.get(name + ' ' + outcome):
        feature_stats[name + ' ' + outcome] = [0.0, 0.0]
      feature_stats[name + ' ' + outcome][0] += value
      feature_stats[name + ' ' + outcome][1] += value * value

  for name in feature_names.keys():
    print name + ":"
    for outcome in count_by_outcome.keys():
      total = count_by_outcome[outcome]
      if (feature_stats.get(name + ' ' + outcome)):
        mean = feature_stats[name + ' ' + outcome][0] / total
        variance = (feature_stats[name + ' ' + outcome][1] / total) - (mean * mean)
        print outcome, ':', mean, variance

def parallel_eval(me, test_data):

  for threshold in drange(0.05, 0.96, 0.05):
    print 'Using ' + str(threshold) + ' as a cutoff:'
    total = 0.0
    true_positives = 0.0
    false_positives = 0.0
    total_positives = 0.0
    for (context, output) in test_data:
      total += 1
      if (output == 'true'):
        total_positives += 1
        if (me.eval(context, 'true') > threshold):
          true_positives += 1
      elif (output == 'false'):
        if (me.eval(context, 'true') > threshold):
          false_positives += 1

    correct = true_positives + total - total_positives - false_positives
    
    accuracy = (correct / total) * 100
    precision = 0.0
    if ((true_positives + false_positives) > 0.0):
      precision = (true_positives / (true_positives + false_positives)) * 100
    recall = 0.0
    if (total_positives > 0.0):
      recall = (true_positives / total_positives) * 100
    f1 = 0.0
    if (precision + recall > 0.0):
      f1 = (2 * precision * recall) / (precision + recall)

    print "\tAccuracy: %2.3f, Precision: %2.3f, Recall: %2.3f, F1: %2.3f" % (accuracy, precision, recall, f1)


def drange(start, stop, step):
  r = start
  while r < stop:
    yield r
    r += step

if __name__ == "__main__":
  main()
