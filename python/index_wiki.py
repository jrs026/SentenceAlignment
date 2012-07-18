#!/usr/bin/python

import re
import sys

from optparse import OptionParser

import wiki_dump

def main():
  parser = OptionParser()

  parser.add_option("-c", "--create_dump", dest="dump_file", default="",
      help="Index a Wikipedia dump (.bz2)")
  parser.add_option("-o", "--output_file", dest="output_file", default="",
      help="Location where the indexed dump will be printed")
  parser.add_option("-i", "--index_file", dest="index_file", default="",
      help="Location of a previously saved index")
  parser.add_option("-d", "--wiki_file", dest="wiki_file", default="",
      help="Location of a previously saved wikitext file")

  (opts, args) = parser.parse_args()

  wd = wiki_dump.WikiDump()
  if opts.dump_file and opts.output_file:
    wd.CreateDump(opts.dump_file, opts.output_file, opts.output_file + '.index')

  if opts.index_file and opts.wiki_file:
    wd.LoadIndex(opts.index_file, opts.wiki_file)
    print "Finished loading index"
    print wd.GetArticle("ppl")
    #for title,wiki_text in wd.IterateArticles():
    #  print title,wiki_text[:50]

if __name__ == "__main__":
  main()
