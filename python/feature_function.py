import math

class FeatureFunction(object):
  """A generic class for extracting features from a sentence pair."""

  def __init__(self, opts):
    self.opts = opts

  def get_features(self, source, target):
    return []

  @staticmethod
  def poisson_prob(mean, observed):
    p = math.exp(-mean)
    for i in xrange(observed):
      p *= mean
      p /= i+1
    return p

class DummyFeature(FeatureFunction):
  """Always extracts a dummy feature with value 1.0."""

  def __init__(self, opts):
    super(DummyFeature, self).__init__(opts)

  def get_features(self, source, target):
    return [('bias', 1.0)]

class LengthFeatures(FeatureFunction):
  """Extracts features only based on the lengths of the two sentences."""

  def __init__(self, opts):
    super(LengthFeatures, self).__init__(opts)

  def get_features(self, source, target):
    source_len = len(source)
    target_len = len(target)
    len_ratio = target_len / (1.0 * source_len)
    poisson_length = self.poisson_prob(source_len, target_len)

    context = []
    context.append(('poisson_length', math.log(poisson_length)))
    context.append(('log_ratio', math.log(len_ratio)))
    return context

class DictionaryFeatures(FeatureFunction):
  """Extracts bag-of-words features after a projection through a bilingual
  dictionary.
  """

  def __init__(self, opts, dictionary):
    super(DictionaryFeatures, self).__init__(opts)
    self.dictionary = dictionary

  def get_features(self, source, target):
    proj_source = self.project(source)
    context = []
    context.append(('cosine_sim', self.cosine_sim(target, proj_source)))
    return context

  def cosine_sim(self, list_vec, hash_vec):
    """Computes the cosine similarity between two vectors represented by a list
    and a hash.
    """
    # Sum of squares:
    list_sos = len(list_vec)
    hash_sos = 0.0
    for key,val in hash_vec.iteritems():
      hash_sos += val**2
    denominator = math.sqrt(list_sos) * math.sqrt(hash_sos)

    numerator = 0.0
    for x in list_vec:
      numerator += hash_vec.get(x, 0.0)

    if denominator > 0.0:
      return numerator / denominator
    else:
      return 0.0

  def project(self, source):
    """Project the source sentence through the dictionary and return the
    bag-of-words vector in the target space.
    """
    proj_source = {}
    for s in source:
      s_proj = self.dictionary.get(s, [])
      for t in s_proj:
        if not t in proj_source:
          proj_source[t] = 0.0
        proj_source[t] += 1.0

    return proj_source

class Model1Features(FeatureFunction):
  """Extracts IBM Model 1 features. Requres a t-table and language model."""

  def __init__(self, opts, m1_probs, lm):
    super(Model1Features, self).__init__(opts)
    self.m1_probs = m1_probs
    self.lm = lm

  def get_features(self, source, target):
    cov_vals = [0.1, 0.25]
    target_cov = {}
    for v in cov_vals:
      target_cov[v] = 0.0
    max_lprob = 0.0
    total_lprob = 0.0
    poisson_length = self.poisson_prob(len(source), len(target))
    lm_prob = math.log(self.poisson_prob(self.lm[0], len(target)))
    for t in target:
      t_score = 0.0
      max_t_score = 0.0
      lm_prob += math.log(self.lm[1].get(t, self.opts.prob_floor))
      for s in source:
        prob = self.m1_probs.get((s, t), self.opts.prob_floor)
        t_score += prob
        if (prob > max_t_score):
          max_t_score = prob

      max_lprob += math.log(max_t_score)
      total_lprob += math.log(t_score / len(source))
      for v in cov_vals:
        if (max_t_score > v):
          target_cov[v] += 1

    context = []

    for v in cov_vals:
      context.append(('target_cov_' + str(v), target_cov[v] / len(target)))

    context.append(('norm_log_target_prob', total_lprob / len(target)))
    context.append(('norm_log_target_max_prob', max_lprob / len(target)))

    context.append(('total_model', total_lprob + math.log(poisson_length) - lm_prob))
    context.append(('norm_total_model', (total_lprob + math.log(poisson_length) -
        lm_prob) / len(target)))

    return context
