#!/usr/bin/python

# Class for parsing Wikipedia markup (wikitext) and returning tokenized/sentence
# split text. It uses wpTextExtractor for removing the wikitext and nltk for
# tokenization.

from collections import deque
import htmlentitydefs
import os
import re
import sys

import codecs
import pickle
import nltk.data
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize.punkt import PunktWordTokenizer

from mwlib import uparser
import mwlib.parser.nodes as nodes

class WikiParser:
  
  # Requires the location of sentence breaking model
  def __init__(self, sbreak_model_file=None):
    if sbreak_model_file:
      self.sbreak_model = nltk.data.load(sbreak_model_file)
    else:
      self.sbreak_model = None

  def ToPlainText(self, wikitext):
    paragraphs = self.WikiToText(wikitext)
    raw_sentences = self.SentenceBreak(paragraphs)
    # The Punkt tokenizer leaves punctuation attached, so fix this:
    sentences = []
    for line in raw_sentences:
      sentences.append(re.sub(r'(\S)([!?.])$', '\\1 \\2', line))
    return sentences

  def SentenceBreak(self, paragraphs):
    output = []
    for paragraph in paragraphs:
      wb_para = ' '.join(PunktWordTokenizer().tokenize(paragraph.strip()))
      if self.sbreak_model:
        output.extend(self.sbreak_model.tokenize(wb_para))
      else:
        output.append(wb_para)
    return output

  def WikiToText(self, wikitext):
    clean_wiki = remove_templates(unescape(wikitext))
    tree = uparser.parseString(title='', raw=clean_wiki)
    ignored_types = ['link']
    ignored_tags = ['ref']
    text_nodes = []
    node_stack = deque([tree])
    while len(node_stack) > 0:
      node = node_stack.popleft()
      # Some nodes don't have a type, not sure what they are
      if hasattr(node, 'type'):
        if isinstance(node.type, (int, long)):
          text_nodes.append(node.asText())
        elif node.type == "link" and node.__class__ == nodes.ArticleLink:
          # mwlib misses some of the interlanguage links
          if ':' not in node.target:
            text_nodes.append(node.target)

        if node.type == 'complex_tag' and node.tagname in ignored_tags:
          continue

        if node.type not in ignored_types:
          children = deque([])
          for c in node.children:
            children.appendleft(c)
          node_stack.extendleft(children)

    full_text = re.sub(r'\n+', '\n', ''.join(text_nodes), re.MULTILINE).strip()
    # Turn the result into lines
    result = []
    for line in full_text.split('\n'):
      clean_line = re.sub('\s+', ' ', line).strip()
      if len(clean_line) > 0:
        result.append(clean_line)

    return result

    return sentences

# Some utility functions
def remove_templates(wikitext):
  old_len = len(wikitext)
  result = re.sub(r'\{(\{|\|)[^{}]*(\}|\|)\}', '', wikitext, re.MULTILINE)
  while len(result) != old_len:
    old_len = len(result)
    result = re.sub(r'\{(\{|\|)[^{}]*(\}|\|)\}', '', result, re.MULTILINE)
  return result

##
# Removes HTML or XML character references and entities from a text string.
#
# @param text The HTML (or XML) source text.
# @return The plain text, as a Unicode string, if necessary.
def unescape(text):
  def fixup(m):
    text = m.group(0)
    if text[:2] == "&#":
      # character reference
      try:
        if text[:3] == "&#x":
          return unichr(int(text[3:-1], 16))
        else:
          return unichr(int(text[2:-1]))
      except ValueError:
        pass
    else:
      # named entity
      try:
        text = unichr(htmlentitydefs.name2codepoint[text[1:-1]])
      except KeyError:
        pass
    return text # leave as is
  return re.sub("&#?\w+;", fixup, text)
