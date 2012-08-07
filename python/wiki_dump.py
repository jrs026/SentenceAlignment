#!/usr/bin/python

# Class for storing an indexed wiki dump

import bz2
import codecs
import re
import sys

class WikiDump:

  def __init__(self):
    # Maps article titles to page ids
    self.title_to_id = {}

    # Maps page ids to a (title, start_offset, length) tuple
    self.id_to_fullinfo = {}

    self.dump_file = ''
    # title, length pairs for reading the wikipedia dump sequentially
    self.page_lengths = []

  def CreateDump(self, dump_file, out_file, out_index_file):
    wiki_out = codecs.open(out_file, encoding='utf-8', mode='w')
    index_out = codecs.open(out_index_file, encoding='utf-8', mode='w')

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
          wiki_out.write(s.decode('utf-8'))
          state = 'text_end'
      elif state == 'text_end':
        m = text_end.match(line)
        if m:
          s = m.group(1) + "\n"
          current_offset += len(s)
          wiki_out.write(s.decode('utf-8'))
          page_length = current_offset - current_start
          self.page_lengths.append((current_title, page_length))
          index_out.write(
              (current_title + "\t" + current_id + "\t"
              + str(current_start) + "\t" + str(page_length) +
              "\n").decode('utf-8')
              )
          self.title_to_id[current_title] = int(current_id)
          self.id_to_fullinfo[int(current_id)] = (
              current_title, int(current_offset), int(page_length)
              )
          state = 'title'
        else:
          current_offset += len(line)
          wiki_out.write(line.decode('utf-8'))
    
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
      self.title_to_id[fields[0]] = int(fields[1])
      self.id_to_fullinfo[int(fields[1])] = (fields[0], int(fields[2]), int(fields[3]))
      self.page_lengths.append((fields[0], int(fields[3])))

  def GetArticle(self, title):
    if title in self.title_to_id:
      page_id = self.title_to_id[title]
      (dummy_title, start, length) = self.id_to_fullinfo[page_id]

      dump = open(self.dump_file, 'r')
      dump.seek(start)
      wiki_text = dump.read(length)
      dump.close()
      return wiki_text.decode('utf-8')
    else:
      return None


  def IterateArticles(self):
    dump = open(self.dump_file, 'r')
    for title,page_length in self.page_lengths:
      yield (title, dump.read(page_length).decode('utf-8'))
    dump.close()

  # Iterate over title pairs for the given language
  def IterateInterwiki(self, interwiki_file, language):
    interwiki = open(interwiki_file, 'r')
    sql_re = re.compile('INSERT INTO `langlinks` VALUES ')
    # Matches page_id, foreign_title
    entry_re = re.compile('\((\d+),\'' + language + '\',\'(.*?)\'\)', re.U)
    for line in interwiki:
      if sql_re.match(line):
        for m in entry_re.finditer(line):
          page_id = int(m.group(1))
          if page_id in self.id_to_fullinfo:
            title = self.id_to_fullinfo[page_id][0]
#            yield (title, m.group(2).decode('utf-8'))
            yield (title, m.group(2))

    interwiki.close()
