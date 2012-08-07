#!/usr/bin/python

import os
import re
import sys

from optparse import OptionParser

from mwlib import uparser
import mwlib.parser.nodes as nodes

import wiki_dump
import wiki_parser

def main():
  cwd = os.getcwd()
  parser = OptionParser()

  parser.add_option("-c", "--create_dump", dest="dump_file", default="",
      help="Index a Wikipedia dump (.bz2)")
  parser.add_option("-o", "--output_file", dest="output_file", default="",
      help="Location where the indexed dump will be printed")
  parser.add_option("-i", "--index_file", dest="index_file", default="",
      help="Location of a previously saved index")
  parser.add_option("-d", "--wiki_file", dest="wiki_file", default="",
      help="Location of a previously saved wikitext file")
  parser.add_option("--inter_wiki", dest="interwiki_file", default="",
      help="Location of an interwiki links SQL file")
  parser.add_option("--iw_out", dest="iw_out", default="",
      help="Print interwiki links to this file")
  parser.add_option("-l", "--language_code", dest="language_code", default="",
      help="Language code of the target Wikipedia of the interwiki links")
  parser.add_option("--dict_trans", dest="dict_trans", default="",
      help="Print Wiktionary entries for the given language.\n" +
        "The language should be specified as a full name.")
  parser.add_option("--dict_trans_out", dest="dict_trans_out", default="",
      help="Dictionary entries are printed here (tab separated)")

  # For creating document pairs
  parser.add_option("--text_out", dest="text_out", default="",
      help="Output cleaned wikitext to this file")
  parser.add_option("--iw_in", dest="iw_in", default="",
      help="Saved interwiki links")

  (opts, args) = parser.parse_args()

  wd = wiki_dump.WikiDump()
  if opts.dump_file and opts.output_file:
    wd.CreateDump(opts.dump_file, opts.output_file, opts.output_file + '.index')

  # Used to identify pages outside of the main namespace
  special_page = re.compile('^\S+:')

  # TODO: Temporary, many things not handled in the options
  if opts.text_out and opts.iw_in:
    print "Writing article pairs from", opts.iw_in, "to", opts.text_out
    source_wp = wiki_parser.WikiParser('old_models/es_model.pickle')
    target_wp = wiki_parser.WikiParser('old_models/en_sbreak.pickle')
    source_dump = wiki_dump.WikiDump()
    source_dump.LoadIndex(cwd + '/data/es_dump.index', cwd + '/data/es_dump')
    print 'Done loading es_dump.index'
    target_dump = wiki_dump.WikiDump()
    target_dump.LoadIndex(cwd + '/data/en_dump.index', cwd + '/data/en_dump')
    print 'Done loading en_dump.index'

    source_out = open(opts.text_out + '.source', 'w')
    target_out = open(opts.text_out + '.target', 'w')
    count = 0
    title_list = open(opts.iw_in, mode='r')
    for line in title_list:
      (target_title, source_title) = line.strip().split('\t')
      if special_page.match(source_title) or special_page.match(target_title):
        continue
      source_wt = source_dump.GetArticle(source_title)
      target_wt = target_dump.GetArticle(target_title)
      if not source_wt or not target_wt:
        continue
      if re.match('^#REDIREC', source_wt, re.IGNORECASE) or re.match('^#REDIREC', target_wt, re.IGNORECASE):
        continue
      source_sents = source_wp.ToPlainText(source_wt)
      if len(source_sents) == 0:
        continue
      target_sents = target_wp.ToPlainText(target_wt)
      if len(target_sents) == 0:
        continue
      print source_title, "\t\t", target_title
      source_out.write('\n'.join(source_sents).encode('utf-8') + '\n\n')
      target_out.write('\n'.join(target_sents).encode('utf-8') + '\n\n')
      count += 1

    print "Wrote", count, "document pairs"
    source_out.close()
    target_out.close()

  if opts.index_file and opts.wiki_file:
    wd.LoadIndex(opts.index_file, opts.wiki_file)

    if opts.interwiki_file and opts.language_code and opts.iw_out:
      iw_file = opts.interwiki_file
      lc = opts.language_code
      iw_out = open(opts.iw_out, 'w')
      for source_title,target_title in wd.IterateInterwiki(iw_file, lc):
        if not special_page.match(source_title) and not special_page.match(target_title):
          iw_out.write(source_title + "\t" + target_title + "\n")
      iw_out.close()

    if opts.dict_trans:
      # The third group will contain the entries
      dict_out = None
      if opts.dict_trans_out:
        dict_out = open(opts.dict_trans_out, 'w')
      dict_line = re.compile(
          r'^\*\s*(\[\[|)' + opts.dict_trans + r'(\]\]|):(.*)$',
          re.IGNORECASE)
      print dict_line.pattern

      # Matches an individual translation entry.
      # Groups:
      #   1: Template type ('+' '-' or '')
      #   2: Language code
      #   3: Translation
      #   4: Rest of the options (TODO)
      dict_entry = re.compile('\{\{t(\+|\-|)\|([^\|\}]+)\|([^\|\}]+)(\|[^\|\}]*)*\}\}')
      print dict_entry.pattern
      
      for title, wiki_text in wd.IterateArticles():
        if special_page.match(title):
          continue
        for line in wiki_text.splitlines():
          line_match = dict_line.search(line)
          if line_match:
            entries = line_match.group(3)
            print entries
            for entry in dict_entry.finditer(entries):
              print "\t", entry.groups()
              if dict_out:
                dict_out.write(title + "\t" + entry.group(3) + "\n")

      if dict_out:
        dict_out.close()

def print_tree(wikitext):
  """Print all of the nodes in the parse tree created from the wikitext."""
  clean_wiki = wiki_parser.remove_templates(wiki_parser.unescape(wikitext))
  tree = uparser.parseString(title='', raw=clean_wiki)
  result = ''
  node_stack = deque([(tree, 0)])
  while len(node_stack) > 0:
    (node, level) = node_stack.popleft()
    node_str = str(node)
    if hasattr(node, 'type'):
      node_str = str(node.type) + ' | ' + str(node)
    result += (level * '\t') + node_str + '\n'
    children = deque([])
    for c in node.children:
      children.appendleft((c, level+1))
    node_stack.extendleft(children)

  return result

if __name__ == "__main__":
  main()
