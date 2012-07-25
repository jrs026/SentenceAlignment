import copy
import os

import maxent

import feature_function
import vocab

class SentencePairExtractor:
  """Contains a classifier for determining whether or not a sentence pair is
  parallel.
  """

  def __init__(self, opts, debug=False):
    """Pass in program options for now."""
    self.opts = opts
    self.source_vocab = vocab.Vocab()
    self.target_vocab = vocab.Vocab()
    self.me = maxent.MaxentModel()
    self.m1_probs = {}
    self.lm = None
    self.dictionary = {}
    self.feature_functions = []
    self.debug = debug

  def init_feature_functions(self):
    """Should be called after loading Model1, language model, or other data."""
    self.feature_functions.append(feature_function.DummyFeature(self.opts))
    self.feature_functions.append(feature_function.LengthFeatures(self.opts))
    if len(self.m1_probs) > 0 and self.lm:
      self.feature_functions.append(feature_function.Model1Features(
          self.opts, self.m1_probs, self.lm))
    if len(self.dictionary) > 0:
      self.feature_functions.append(feature_function.DictionaryFeatures(
          self.opts, self.dictionary))

  def get_features(self, source_sent, target_sent):
    """Return a featurized context for a sentence pair based on all feature
    functions.
    """
    if self.debug:
      source_words = []
      for s in source_sent:
        source_words.append(self.source_vocab.id_lookup(s))
      target_words = []
      for t in target_sent:
        target_words.append(self.target_vocab.id_lookup(t))
      print "Source:", ' '.join(source_words)
      print "Target:", ' '.join(target_words)
    context = []
    for ff in self.feature_functions:
      context.extend(ff.get_features(source_sent, target_sent))
    return context

  def read_parallel_data(self, source_file, target_file):
    """Read parallel data and update the vocabularies."""
    source_parallel = self.read_sentences(source_file, self.source_vocab)
    target_parallel = self.read_sentences(target_file, self.target_vocab)
    return (source_parallel, target_parallel)

  def read_comp_data(self, base_filename):
    """Read comparable document pairs (which may be annotated)."""
    source_docs = self.read_docs(base_filename + ".source", self.source_vocab)
    target_docs = self.read_docs(base_filename + ".target", self.target_vocab)
    if os.path.exists(base_filename + ".alignment"):
      alignment_docs = self.read_docs(base_filename + ".alignment")
      return (source_docs, target_docs, alignment_docs)
    else:
      return (source_docs, target_docs)

  def train_model(self, training_data):
    self.me.begin_add_event()
    for example in training_data:
      self.me.add_event(example[0], example[1])
    self.me.end_add_event()
    self.me.train(self.opts.max_iterations, "lbfgs", self.opts.l2_norm)

  def test_model(self, test_data, threshold=0.5):
    """Return accuracy, precision, recall, and f1 using the given classification
    threshold.
    """
    total = 0.0
    true_positives = 0.0
    false_positives = 0.0
    total_positives = 0.0
    for (context, output) in test_data:
      total += 1
      if (output == 'true'):
        total_positives += 1
        if (self.me.eval(context, 'true') > threshold):
          true_positives += 1
      elif (output == 'false'):
        if (self.me.eval(context, 'true') > threshold):
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

    return (accuracy, precision, recall, f1)
    
  def extract_sentences(self, raw_source, raw_target, out_file, threshold=0.5):
    """Extract parallel sentences from source and target comparable
    documents.
    """
    s_out = open(out_file + '.source', 'w')
    t_out = open(out_file + '.target', 'w')
    for i in xrange(0, len(raw_source)):
      for s_sent in raw_source[i]:
        for t_sent in raw_target[i]:
          len_ratio = len(t_sent) / (1.0 * len(s_sent))
          if (len_ratio < 1.0):
            len_ratio = 1.0 / len_ratio
          if (len_ratio < self.opts.max_len_ratio):
            context = get_features(s_sent, t_sent)
            if (self.me.eval(context, 'true') > threshold):
              source_words = []
              for s in s_sent:
                source_words.append(self.source_vocab.id_lookup(s))
              target_words = []
              for t in t_sent:
                target_words.append(self.target_vocab.id_lookup(t))
              s_out.write(' '.join(source_words) + "\n")
              t_out.write(' '.join(target_words) + "\n")

    s_out.close()
    t_out.close()

  def create_annotated_parallel_data(self, source_sents, target_sents):
    """Create annotated sentence pair instances from parallel data. This will
    automatically create a train/test split and return them separately.
    """
    train_data = []
    test_data = []
    for i,source in enumerate(source_sents):
      if (i < self.opts.test_max):
        for j in range(0, self.opts.test_max):
          context = self.get_features(source, target_sents[j])
          if (j == i):
            test_data.append((context, 'true'))
          else:
            test_data.append((context, 'false'))
      else:
        neg_examples = 0
        for j in range(i - self.opts.example_window, i + self.opts.example_window + 1):
          if ((j >= self.opts.test_max) and (j < len(target_sents)) and (j != i)):
            len_ratio = len(target_sents[j]) / (1.0 * len(source))
            if (len_ratio < 1.0):
              len_ratio = 1.0 / len_ratio
            if (len_ratio < self.opts.max_len_ratio):
              neg_examples += 1
              context = self.get_features(source, target_sents[j])
              train_data.append((context, 'false'))

        true_context = self.get_features(source, target_sents[i])
        train_data.append((true_context, 'true'))

    return (train_data, test_data)


  def create_annotated_comp_data(self, source_docs, target_docs, alignments):
    """Create annotated sentence pair instances from comparable data."""
    # Read the alignments into a dict for each document
    a_dicts = []
    for a in alignments:
      a_dict = {}
      for pair in a:
        (s, t) = pair.split()
        a_dict[(int(s), int(t))] = 1.0
      a_dicts.append(a_dict)

    annotated_data = []
    for i in xrange(0, len(source_docs)):
      source_sents = source_docs[i]
      target_sents = target_docs[i]
      a_dict = a_dicts[i]
      for s,s_sent in enumerate(source_sents):
        for t,t_sent in enumerate(target_sents):
          len_ratio = len(t_sent) / (1.0 * len(s_sent))
          if (len_ratio < 1.0):
            len_ratio = 1.0 / len_ratio
          if (len_ratio < self.opts.max_len_ratio):
            outcome = 'false'
            if (s, t) in a_dict:
              outcome = 'true'
            context = self.get_features(s_sent, t_sent)
            annotated_data.append((context, outcome))
      
    return annotated_data

  def create_lm(self, sentences):
    """Create a unigram language model from the given sentences."""
    total_words = 0.0
    word_probs = {}
    for sentence in sentences:
      total_words += len(sentence)
      for word in sentence:
        if not word_probs.get(word):
          word_probs[word] = 0.0
        word_probs[word] += 1.0

    length_mean = total_words / len(sentences)
    for word in word_probs.keys():
      word_probs[word] /= total_words
      if word_probs[word] < self.opts.prob_floor:
        word_probs[word] = self.opts.prob_floor
    self.lm = (length_mean, word_probs)
    return self.lm

  def read_m1_probs(self, filename):
    """Read Model 1 probabilities (t-table)."""
    f = open(filename)
    for line in f:
      (s, t, cost) = line.strip().split()
      s_index = self.source_vocab.add_word(s)
      t_index = self.target_vocab.add_word(t)
      self.m1_probs[(s_index, t_index)] = float(cost)
    f.close()

  def read_dictionary(self, filename):
    """Read a dictionary file with tab separated entries. For now, we are
    ignoring multiword entries.
    """
    f = open(filename)
    for line in f:
      (s_word, t_word) = line.strip().split("\t")
      if ' ' in s_word or ' ' in t_word:
        continue
      s = self.source_vocab.add_word(s_word)
      t = self.target_vocab.add_word(t_word)
      if not s in self.dictionary:
        self.dictionary[s] = []
      self.dictionary[s].append(t)
    f.close()

  def dict_from_m1(self, filename, cutoff=0.1):
    """Read in dictionary entries from a Model 1 t-table file. If the t-table
    probability is above the cutoff, then it is added to the dictionary.
    """
    f = open(filename)
    for line in f:
      (s_word, t_word, cost) = line.strip().split()
      if float(cost) > cutoff:
        s = self.source_vocab.add_word(s_word)
        t = self.target_vocab.add_word(t_word)
        if not s in self.dictionary:
          self.dictionary[s] = []
        self.dictionary[s].append(t)
    f.close()

  @staticmethod
  def read_docs(filename, vocab=None):
    """Read documents (or alignments) separated by blank lines"""
    docs = []
    current_doc = []
    for line in file(filename):
      if len(line.strip()) == 0:
        if len(current_doc) > 0:
          docs.append(copy.deepcopy(current_doc))
          current_doc = []
      else:
        if vocab:
          current_sent = []
          for token in line.split():
            current_sent.append(vocab.add_word(token))
          current_doc.append(current_sent)
        else:
          current_doc.append(line.strip())

    if len(current_doc) > 0:
      docs.append(current_doc)
    return docs

  @staticmethod
  def read_sentences(filename, vocab):
    """Read text from the file into arrays of vocabulary entries."""
    sents = []
    f = open(filename)
    for line in f:
      current_sent = []
      for token in line.split():
        current_sent.append(vocab.add_word(token))
      sents.append(current_sent)
    f.close()
    return sents
