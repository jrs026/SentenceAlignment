#!/usr/bin/python

import copy
import math
import os
import re
import sys

from optparse import OptionParser

import maxent

def main():
  parser = OptionParser()

  parser.add_option("-p", "--parallel_data", dest="training_file",
      default="../data/euro_esen_10k",
      help="Parallel data, expecting \".source\" and \".target\"")

  parser.add_option("-c", "--comparable_data", dest="comp_data",
      default="../data/es_dev",
      help="Annotated comparable data, expecting \".source\", \".target\", and \".alignment\"")

  parser.add_option("-r", "--raw-data", dest="raw_data",
      default="../data/esen_docs_small",
      help="Raw comparable data, expecting \".source\" and \".target\"")

  parser.add_option("-t", "--t-table", dest="m1_data", default="small.model",
      help="Word alignment parameters from some parallel data")

  parser.add_option("-e", "--example-window", dest="example_window", type="int",
      default=3, help="Size of the example window for gathering training data")

  parser.add_option("--length-ratio", type="float", dest="max_len_ratio",
      default=3.0, 
      help="Maximum length ratio for sentences to be considered parallel")

  parser.add_option("--test-max", type="int", dest="test_max", default=100,
      help="Number of sentences from the parallel data to use as test data")

  parser.add_option("--prob-floor", type="float", dest="prob_floor",
      default=1e-4, help="Lowest probability value for LM and M1")

  parser.add_option("--l2-norm", type="float", dest="l2_norm", default="2.0",
      help="L2 normalizing value for the Maxent model")

  (opts, args) = parser.parse_args()

  source_parallel = read_lines(opts.training_file + '.source')
  target_parallel = read_lines(opts.training_file + '.target')
  (source_docs, target_docs, alignments) = read_comp_data(opts.comp_data)
  (raw_source, raw_target) = read_comp_data(opts.raw_data)

  t_lm = create_lm(target_parallel, opts)

  m1 = read_m1_data(opts.m1_data)

  #(train_data, test_data) = create_train_test_data(
  #    me, source_parallel, target_parallel, m1, t_lm, opts)
  comp_data = create_comp_test_data(
      source_docs, target_docs, alignments, m1, t_lm, opts)
  print_feature_stats(comp_data)
  folds = range(4)
  for fold in folds:
    comp_test_data = []
    me = maxent.MaxentModel()
  
    print "\nFold " + str(fold+1) + ":"
    me.begin_add_event()
    for i,event in enumerate(comp_data):
      if i % len(folds) == fold:
        comp_test_data.append(event)
      else:
        me.add_event(event[0], event[1])
    me.end_add_event()
    me.train(50, "lbfgs", opts.l2_norm)
    parallel_eval(me, comp_test_data)
    
  # TODO put into opts
  extract = False
  if extract:
    full_me = maxent.MaxentModel()
    full_me.begin_add_event()
    for event in comp_data:
      full_me.add_event(event[0], event[1])
    full_me.end_add_event()
    full_me.train(50, "lbfgs", opts.l2_norm)

    for threshold in drange(0.05, 0.96, 0.05):
      out_file = "data/esen_out_" + str(threshold)
      extract_sentences(full_me, raw_source, raw_target, out_file, threshold,
          m1, t_lm, opts)

  #print me
  #parallel_eval(me, comp_test_data)
  #parallel_eval(me, test_data)

def extract_sentences(me, raw_source, raw_target, out_file, threshold, 
    m1_data, t_lm, opts):
  s_out = open(out_file + '.source', 'w')
  t_out = open(out_file + '.target', 'w')
  for i in xrange(0, len(raw_source)):
    for s_sent in raw_source[i]:
      s_len = len(s_sent.split())
      for t_sent in raw_target[i]:
        t_len = len(t_sent.split())
        len_ratio = t_len / (1.0 * s_len)
        if (len_ratio < 1.0):
          len_ratio = 1.0 / len_ratio
        if (len_ratio < opts.max_len_ratio):
          context = get_features(s_sent, t_sent, m1_data, t_lm, opts)
          if (me.eval(context, 'true') > threshold):
            s_out.write(s_sent + "\n")
            t_out.write(t_sent + "\n")

  s_out.close()
  t_out.close()

def create_lm(sentences, opts):
  total_words = 0.0
  word_probs = {}
  for sentence in sentences:
    words = sentence.split()
    total_words += len(words)
    for word in words:
      if not word_probs.get(word):
        word_probs[word] = 0.0
      word_probs[word] += 1.0

  length_mean = total_words / len(sentences)
  for word in word_probs.keys():
    word_probs[word] /= total_words
    if word_probs[word] < opts.prob_floor:
      word_probs[word] = opts.prob_floor
  return (length_mean, word_probs)

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

def create_train_test_data(me, source_parallel, target_parallel, m1_data, t_lm,
    opts):
  train_data = []
  test_data = []
  for i,source in enumerate(source_parallel):
    if (i < opts.test_max):
      for j in range(0, opts.test_max):
        context = get_features(source, target_parallel[j], m1_data, t_lm, opts)
        if (j == i):
          test_data.append((context, 'true'))
        else:
          test_data.append((context, 'false'))
    else:
      neg_examples = 0
      for j in range(i - opts.example_window, i + opts.example_window + 1):
        if ((j >= opts.test_max) and (j < len(target_parallel)) and (j != i)):
          source_len = len(source.split())
          target_len = len(target_parallel[j].split())
          len_ratio = target_len / (1.0 * source_len)
          if (len_ratio < 1.0):
            len_ratio = 1.0 / len_ratio
          if (len_ratio < opts.max_len_ratio):
            neg_examples += 1
            context = get_features(
                source, target_parallel[j], m1_data, t_lm, opts)
            train_data.append((context, 'false'))
            me.add_event(context, 'false')

      true_context = get_features(
          source, target_parallel[i], m1_data, t_lm, opts)
      train_data.append((true_context, 'true'))
      me.add_event(true_context, 'true', neg_examples)

  return (train_data, test_data)

def create_comp_test_data(source_docs, target_docs, alignments,
    m1_data, t_lm, opts):

  # Read the alignments into a dict for each document
  a_dicts = []
  for a in alignments:
    a_dict = {}
    for pair in a:
      a_dict[' '.join(pair.split())] = 1.0
    a_dicts.append(a_dict)

  test_data = []
  for i in xrange(0, len(source_docs)):
    source_sents = source_docs[i]
    target_sents = target_docs[i]
    a_dict = a_dicts[i]
    for s,s_sent in enumerate(source_sents):
      for t, t_sent in enumerate(target_sents):
        s_len = len(s_sent.split())
        t_len = len(t_sent.split())
        len_ratio = t_len / (1.0 * s_len)
        if (len_ratio < 1.0):
          len_ratio = 1.0 / len_ratio
        if (len_ratio < opts.max_len_ratio):
          outcome = 'false'
          if str(s) + ' ' + str(t) in a_dict:
            outcome = 'true'
          context = get_features(s_sent, t_sent, m1_data, t_lm, opts)
          test_data.append((context, outcome))
    
  return test_data

def get_features(source, target, m1_data, t_lm, opts):
  source_words = source.split()
  target_words = target.split()
  source_len = len(source_words)
  target_len = len(target_words)
  len_ratio = target_len / (1.0 * source_len)
  poisson_length = poisson_prob(source_len, target_len)
  
  # Model 1 features
  #cov_vals = [0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
  cov_vals = [0.1, 0.25]
  target_cov = {}
  for v in cov_vals:
    target_cov[v] = 0.0
  max_lprob = 0.0
  total_lprob = 0.0
  lm_prob = math.log(poisson_prob(t_lm[0], target_len))
  for t in target_words:
    t_score = 0.0
    max_t_score = 0.0
    lm_prob += math.log(t_lm[1].get(t, opts.prob_floor))
    for s in source_words:
      prob = m1_data.get(s + ' ' + t, opts.prob_floor)
      t_score += prob
      if (prob > max_t_score):
        max_t_score = prob

    max_lprob += math.log(max_t_score)
    total_lprob += math.log(t_score / len(source_words))
    for v in cov_vals:
      if (max_t_score > v):
        target_cov[v] += 1

  context = []
  context.append(('bias', 1.0))
  context.append(('poisson_length', math.log(poisson_length) -
      math.log(poisson_prob(t_lm[0], target_len))))
  context.append(('log_ratio', math.log(len_ratio)))

  for v in cov_vals:
    context.append(('target_cov_' + str(v), target_cov[v] / target_len))

#  context.append(('target_score', total_score / target_len))
#  context.append(('target_max_score', max_score / target_len))

  context.append(('norm_log_target_prob', total_lprob / target_len))
  context.append(('norm_log_target_max_prob', max_lprob / target_len))

  context.append(('total_model', total_lprob + math.log(poisson_length) - lm_prob))
  context.append(('norm_total_model', (total_lprob + math.log(poisson_length) -
      lm_prob) / target_len))

  return context

def read_comp_data(filename):
  source_docs = read_docs(filename + ".source")
  target_docs = read_docs(filename + ".target")
  if os.path.exists(filename + ".alignment"):
    alignment_docs = read_docs(filename + ".alignment")
    return (source_docs, target_docs, alignment_docs)
  else:
    return (source_docs, target_docs)

# Read documents (or alignments) separated by blank lines
def read_docs(filename):
  docs = []
  current_doc = []
  for line in file(filename):
    if len(line.strip()) == 0:
      if len(current_doc) > 0:
        docs.append(copy.deepcopy(current_doc))
        current_doc = []
    else:
      current_doc.append(line.strip().lower())

  if len(current_doc) > 0:
    docs.append(current_doc)
  return docs

def read_m1_data(filename):
  m1_data = {}
  f = open(filename)
  for line in f:
    (s, t, cost) = line.split()
    m1_data[s + ' ' + t] = float(cost)

  return m1_data

def read_lines(filename):
  f = open(filename)
  lines = f.readlines()
  f.close()
  return lines
  
def poisson_prob(mean, actual):
  p = math.exp(-mean)
  for i in xrange(actual):
    p *= mean
    p /= i+1
  return p

def drange(start, stop, step):
  r = start
  while r < stop:
    yield r
    r += step

if __name__ == "__main__":
  main()
