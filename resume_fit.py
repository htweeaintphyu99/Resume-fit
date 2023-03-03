# !pip install pdfminer.six
# !pip install tika
# !pip install textract

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string
import gensim
import operator
import re
import os
from gensim import corpora
from gensim.similarities import MatrixSimilarity
from operator import itemgetter
from tika import parser
import time
import textract
from collections import defaultdict
from gensim import corpora
from gensim import models
from gensim import similarities
import sys


# get the start time
# st = time.time()

jd_dir=sys.argv[1] #path of job description file
cv_dir = len(sys.argv[2]) #list of cv paths
resumes_list = sys.argv[2][0:cv_dir]
resumes_list = resumes_list.split(',') # eg: ['cv\Software Engineer.pdf','cv\software-engineer-resume.pdf'] (No space is allowed after comma in each item of the  list)



def get_data(path):
  data = {'File Name': [],'CV': []}
  for i in range(len(resumes_list)):
    try:
      text = parser.from_file(resumes_list[i])
      text = text["content"]
      data['CV'].append(text)
      data['File Name'].append((resumes_list[i].split('/'))[-1])
    except:
      pass

  return data

def read_txt_file(jd_dir):
    with open(jd_dir,encoding="ISO-8859-1") as f:
        contents = f.read()
        return contents

#When we give two inputs such as job description and CVs, the function will output CVs with respective scores that are the most fit with job description
def resume_fit(search_term):
  cv_df = get_data(cv_dir)
  doc_names = cv_df['File Name']
  documents = cv_df['CV']

  # remove common words and tokenize
  stoplist = set('for a of the and to in'.split())
  texts = [[word for word in document.lower().split() if word not in stoplist]for document in documents]

  # remove words that appear only once
  frequency = defaultdict(int)
  for text in texts:
      for token in text:
          frequency[token] += 1

  texts = [
      [token for token in text if frequency[token] > 1]
      for text in texts
  ]

  dictionary = corpora.Dictionary(texts)
  corpus = [dictionary.doc2bow(text) for text in texts]

  lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=100)

  # doc = "Responsibilities Prepare and review compensation and benefits packages Administer health and life insurance programs Implement training and development plans Plan quarterly and annual performance review sessions Inform employees about additional benefits they’re eligible for (e.g extra vacation days) Update employee records with new hire information and/or changes in employment status Maintain organizational charts and detailed job descriptions along with salary records Forecast hiring needs and ensure recruitment process runs smoothly Develop and implement HR policies throughout the organization Monitor budgets by department Process employees’ queries and respond in a timely manner Stay up-to-date and comply with changes in labor legislation Requirements and skills Proven work experience as an HR Specialist or HR Generalist Hands-on experience with Human Resources Information Systems (HRIS), like BambooHR and PeopleSoft Knowledge of Applicant Tracking Systems Solid understanding of labor legislation and payroll process Familiarity with full cycle recruiting Excellent verbal and written communication skills Good problem-solving abilities Team management skills BSc/MSc in Human Resources or relevant field"
  vec_bow = dictionary.doc2bow((read_txt_file(jd_dir)).lower().split())
  vec_lsi = lsi[vec_bow]  # convert the query to LSI space

  index = similarities.MatrixSimilarity(lsi[corpus])  # transform corpus to LSI space and index it


  sims = index[vec_lsi]  # perform a similarity query against the corpus

  resume_match={}

  sims = sorted(enumerate(sims), key=lambda item: -item[1])
  for doc_position, doc_score in sims:
      # print(doc_score, doc_names[doc_position])
      # if round((doc_score* 100),2) >= 0.00:
      #       resume_match[doc_names[doc_position]]=round((doc_score * 100),2)

      # if round((doc_score * 100),2)<=0.00:
      #   resume_match[doc_names[doc_position]]=0.00

      # else: 
      #   resume_match[doc_names[doc_position]]=round((doc_score * 100),2)
      resume_match[doc_names[doc_position]]=round((doc_score * 100),2)
  return resume_match
     # et = time.time()
# get the execution time
# elapsed_time = et - st
# print('Execution time:', elapsed_time, 'seconds')
resume_fit(jd_dir)

