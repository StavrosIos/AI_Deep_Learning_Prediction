
"""
This script is used to process the raw data text files into a format that can be used by the scripts.

These data files reside inside the raw data directory.
"""

import os     
import re
from collections import Counter
import numpy as np





def load_all_articles(data_directory):
    #load all bbc articles and return one very long list of words
    text_data_return=[]
    #grab the names of the files listed in the target directory 
    #loop throug each file, open it and add the words to the resulting list 

    text_files=os.listdir(data_directory)
    #debug=text_files[:190]
    #debug2=text_files[190:197]
    #debug3=text_files[197:199]
    
    for text_file in text_files:
          with open(data_directory+"/"+text_file,'r',encoding='utf8') as f: #open the files
              
                
              text=f.read()
              # remove newlines, clean up the text slightly 
              text = text.replace('\n', ' ')
              text = re.sub(r'\s+', " ", text) # type(text)-> string 
              
              #maybe make them in lower case
      
              
              #break string down to individual words        
              #add a whitespace after every word inside the list
              text_list=text.split()
              text_list=fix_punctuation(text_list)
              #for i in range (0,len(text_list)):
              #    text_list.insert(i*2," ")
              
                  
              #text_data_return.
             # print(text_list)
              #text_data_return.extend(text) function for Char-level analysis
              text_data_return.extend(text_list) # list of words
              
              #print(text_data_return)
              
    return text_data_return      





def save_all_processed_articles(data,path):
      
      #save the processed data (array), to a text files
    with open(path, 'w',encoding='utf-8') as f:
          for item in data:
              f.write("%s\n" % item)
        
    return True    




def fix_punctuation(data_list):
      # takes list, returns a list with fixed punctuation. quotes, fullstops, commas, questionmarks become individual tokens 
      f_list=[]
      chars=[',',"?",".","!","!!","!!!"] 
      
      for seq in data_list:
            lis_seq=list(seq)
            #print(lis_seq)
            
            #if lis_seq[0] == chars[-1]: # if opening quote is found
            #      f_list.append('"')
            #      del lis_seq[0]
                  
            #closing_quote=False   
            
            #if lis_seq[-1] == chars[-1]:
            #      closing_quote=True
            #      del lis_seq[-1]
                  
            char_index=None
            for index in range(len(chars)):
                  if chars[index] == lis_seq[-1]:
                        char_index=index
                  
            
            if char_index is not None: #special char found
                  del lis_seq[-1] # delete the special character at the end.
                  f_list.append(''.join(lis_seq))
                  f_list.append(chars[char_index])
                  #if closing_quote:
                  #      f_list.append('"')
            
            else:
                  f_list.append(''.join(lis_seq))
      
      
      
      
      return f_list


      




### load all data files 

files_data=load_all_articles('raw data/BBC_tech')
token_frequency = Counter(files_data) #weed out bad tokens
print(len(files_data))

save_all_processed_articles(files_data,"processed/bbc_tech_new.txt")




















