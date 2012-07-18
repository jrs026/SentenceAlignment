#!/usr/bin/python

# Class for storing an indexed wiki dump

import bz2
import re
import sys

class WikiDump:

  def __init__(self):
    self.title_index = {}
    self.dump_file = ''
    # title, length pairs for reading the wikipedia dump sequentially
    self.page_lengths = []

  def CreateDump(self, dump_file, out_file, out_index_file):
    wiki_out = open(out_file, mode='w')
    index_out = open(out_index_file, mode='w')

    title = re.compile(r"^\s*<title>(.*)</title>")
    page_id = re.compile(r"    <id>(.*)</id>")
    text_start = re.compile(r"^\s*<text xml:space=\"preserve\">(.*)$")
    text_end = re.compile(r"^(.*)</text>$")

    (current_title, current_id, current_start, current_offset) = ('', '', 0, 0)

    state = 'title' # title, id, text_start, text_end

    dump = bz2.BZ2File(dump_file, 'r')
    line = dump.readline()
    while (line):
      if state == 'title':
        m = title.match(line)
        if m:
          current_title = m.group(1)
          state = 'id'
      elif state == 'id':
        m = page_id.match(line)
        if m:
          current_id = m.group(1)
          state = 'text_start'
      elif state == 'text_start':
        m = text_start.match(line)
        if m:
          current_start = current_offset
          s = m.group(1) + "\n"
          current_offset += len(s)
          wiki_out.write(s)
          state = 'text_end'
      elif state == 'text_end':
        m = text_end.match(line)
        if m:
          s = m.group(1) + "\n"
          current_offset += len(s)
          wiki_out.write(s)
          page_length = current_offset - current_start
          self.page_lengths.append((current_title, page_length))
          index_out.write(current_title + "\t" + current_id + "\t"
              + str(current_start) + "\t" + str(page_length) + "\n")
          state = 'title'
          self.title_index[current_title] = (
              int(current_id), int(current_offset), int(page_length)
              )
          state = 'title'
        else:
          current_offset += len(line)
          wiki_out.write(line)
    
      line = dump.readline()

    wiki_out.close()
    self.dump_file = wiki_out
    index_out.close()

  def LoadIndex(self, index_file, dump_file):
    self.title_index = {}
    self.page_lengths = []
    self.dump_file = dump_file
    for line in file(index_file): 
      fields = line.split("\t")
      self.title_index[fields[0]] = (int(fields[1]), int(fields[2]), int(fields[3]))
      self.page_lengths.append((fields[0], int(fields[3])))

  def GetArticle(self, title):
    wiki_text = None
    if title in self.title_index:
      (page_id, start, length) = self.title_index[title]
      dump = open(self.dump_file, 'r')
      dump.seek(start)
      wiki_text = dump.read(length)
      dump.close()

    return wiki_text

  def IterateArticles(self):
    dump = open(self.dump_file, 'r')
    for title,page_length in self.page_lengths:
      yield (title, dump.read(page_length))
    dump.close()
