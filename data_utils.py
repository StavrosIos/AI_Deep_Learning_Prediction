"""
This script provies the data interface that turns data into usable input to the Neural Network models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re
np.random.seed(0)


def load_processed_data_and_create_vocabulary_of_characters(data_path,batch_size,sequence_size):
      
    print("Loading and processing data") 
    
    #load from text file source
    # list holding one element, only 
    loaded_text=load_processed_articles_by_character(data_path)


    #build up vocabulary from the text 
    vocabulary = sorted(set(loaded_text[0])) 
    print('Vocabulary size {}'.format(len(vocabulary)))
    

    #token_frequency = Counter(loaded_list) 
    #vocabulary = sorted(token_frequency, key=token_frequency.get, reverse=True)

    #dictionaries to map word/tokens to integers
    int_to_vocabulary = {k: w for k, w in enumerate(vocabulary)}
    vocabulary_to_int = {w: k for k, w in int_to_vocabulary.items()}

    #print('Vocabulary size', len(int_to_vocabulary)) #size of the vocabulary, probability over this vector to be predicted

    #text=''.join(loaded_list)

    # text encoded as integers

    text_econded_arr = np.array([vocabulary_to_int[token] for token in loaded_text[0]],dtype=np.int16)
    
    """
    print('Full text size', text_econded_arr.shape)

    print('\n'+"Example conversion-deconversion")
    print(''.join(int_to_vocabulary[index] for index in text_econded_arr[:20]), '== Encoding ==> ', text_econded_arr[:20])
    print('\n')
    print(text_econded_arr[20:35], ' == Reverse  ==> ', ''.join(int_to_vocabulary[index] for index in text_econded_arr[20:35]))

    """

    training_data, labels_data, num_batches = reshape_text_data(text_econded_arr,batch_size,sequence_size)
    


    return training_data, labels_data, num_batches, int_to_vocabulary, vocabulary_to_int




def load_processed_articles_by_character(path):
      
      all_text=[]
      
      #open file,  in line-by-line format it was saved as 
      text = open(path,"r").read() 
      
      #remove new lines
      text= text.replace('\n',' ')
      
      text = re.sub(r'\s+'," ",text) # create one huge string with all the text
      
      all_text.append(text)
      

      return all_text





def load_processed_articles(path):
      
      # read processed text data
      text_data=[]
      with open(path,'r',encoding='utf8') as f: #open the file
            
            text=f.read() #READ LINE
            text = text.replace('\n', ' ') # delete empty rows

            text_list=text.split()
            for i in range (0,len(text_list)):
                  text_list.insert(i*2," ")
                  
            text_data.extend(text_list) # list of words

      return text_data



def load_processed_data_and_create_vocabulary(data_path,batch_size,sequence_size):
      
    print("Loading and processing data") 
    
    #example--> datasets/processed/bbc_tech.txt
    loaded_list=load_processed_articles(data_path)

    token_frequency = Counter(loaded_list) 
    vocabulary = sorted(token_frequency, key=token_frequency.get, reverse=True)

    #dictionaries to map word/tokens to integers
    int_to_vocabulary = {k: w for k, w in enumerate(vocabulary)}
    vocabulary_to_int = {w: k for k, w in int_to_vocabulary.items()}

    print('Vocabulary size', len(int_to_vocabulary)) #size of the vocabulary, probability over this vector to be predicted

    #text=''.join(loaded_list)

    # text encoded as integers

    text_econded_arr = np.array([vocabulary_to_int[token] for token in loaded_list],dtype=np.int16)
    print('Full text size', text_econded_arr.shape)

    print('\n'+"Example conversion-deconversion")
    print(''.join(int_to_vocabulary[index] for index in text_econded_arr[:20]), '== Encoding ==> ', text_econded_arr[:20])
    print('\n')
    print(text_econded_arr[20:35], ' == Reverse  ==> ', ''.join(int_to_vocabulary[index] for index in text_econded_arr[20:35]))



    training_data, labels_data, num_batches = reshape_text_data(text_econded_arr,batch_size,sequence_size)
    


    return training_data, labels_data, num_batches, int_to_vocabulary, vocabulary_to_int





def reshape_text_data(text_econded_arr,batch_size,sequence_size,):
    #take in encoded text as integers and reshape it 
    ##### Split text data into train and target pools 
    
    sequence_size=sequence_size # the size of the sequence -->  50,100, 200 tokens
    batch_size=batch_size  # number of sequences each batch holds   -->  64,32,16 sample/sequences
    

    #number of batches should be 68
    num_batches=int((len(text_econded_arr) / (sequence_size)) / (batch_size ))
    #print(len(loaded_list))

    
    training_text = text_econded_arr[ : num_batches * batch_size * sequence_size]
    output_text = text_econded_arr[1 : num_batches * batch_size * sequence_size+1]

    training_data = np.split(training_text, batch_size)
    target_data = np.split(output_text, batch_size)

    #for the correct dimensionality stack 
    #text_in==> training data pool, text_out==>target pool
    training_text = np.stack(training_data)
    label_text = np.stack(target_data)

    return training_text, label_text, num_batches





def save_generated_text(text,path,iteration,dataset,method='TOPK'):
      
    #break the text in 3 pieces and save each one with a newline  in  between 
    end_boundary=len(text)
    newline1=int(end_boundary/2)
    newline2=int(newline1/2)

      #save the processed data (array), to a text files
    with open(path+'/generated article'+" Iter_"+str(iteration)+' '+str(dataset)+'.txt', 'w',encoding='utf-8') as f:
           # for item in text.split():
           f.write("Generating method used: "+method)
           f.write("\n")
           f.write(text[:newline2])
           f.write("\n")
           f.write(text[newline2:newline1])
           f.write("\n")
           f.write(text[newline1:end_boundary])
    
    print("\n")    
    print("Generated article saved")
    print("\n")
    
    
    return True    





def generate_data_minibatch(training_text, output_text, seq_size, num_of_batches):
    for index in range(0, num_of_batches * seq_size, seq_size):
        yield training_text[:, index:index+seq_size], output_text[:, index:index+seq_size]
        

