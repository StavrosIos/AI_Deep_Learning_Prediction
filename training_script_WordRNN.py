"""

This script will be used to train a Word-level RNN for generating text.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import data_utils as utils
import nn_models
import time
import pickle

"""
Load data
Train network - word-level RNN
Save generated text
Save network 

Generating text methods---> "TOPK" is GREEDY SAMPLING. Where we sample from top x most probable words
Generating text ---> "MULTINOMIAL" distribution over the vocabulary. The logits have been transformed using tensor.exp()
"""

torch.manual_seed(0) #setting seed for reproduucible results
np.random.seed(0)


#processed data path, data file to load 
file_path="datasets/processed/bbc_sport.txt"

sequence_size=100
batch_size=64

training_text, output_text,num_of_batches, int_to_vocab, vocab_to_int = utils.load_processed_data_and_create_vocabulary(file_path,
                                                                                                                        batch_size=batch_size,
                                                                                                                        sequence_size=sequence_size)


print('\n')
print("Training agent")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


criterion = nn.CrossEntropyLoss()
embedding_layer_size = 512
lstm_size=100 


#create Recurrent Neural Net
LSTM = nn_models.LSTM(vocabulary_size=len(int_to_vocab), embedding_dimensionality=embedding_layer_size, lstm_size=lstm_size)
LSTM = LSTM.to(device) #put RNN on Cuda GPU
LSTM.train()
LSTM.cuda()


optimizer = torch.optim.Adam(LSTM.parameters(), lr=0.001)
iteration = 0
training_losses=[]
epochs= 500
generate_text = 5_000 # generate text and save model after x iterations
save_loss = 100 #save loss after x iterations
iteration=0
start_train = time.time()

for epoch in range(1,epochs+1):

    
    states = nn_models.initialize_zero_state(device,batch_size,lstm_size) #Initialize RNN state for each epoch as 0
    
    batches = utils.generate_data_minibatch(training_text, output_text, 
                                            seq_size=sequence_size, 
                                            num_of_batches=num_of_batches)
    
    for mini_batch_x, mini_batch_y in batches:
        
        iteration+=1
          
          
        # Create tensors and put on the CUDA device
        mini_batch_x = torch.tensor(mini_batch_x).to(device, dtype=torch.long)
        mini_batch_y = torch.tensor(mini_batch_y).to(device, dtype=torch.long)   
                
        
        #detach states before feeding them to the network  

        predictions, states = LSTM(mini_batch_x, states)
        
        states = nn_models.detach(states)        
        
        
        # need correct shape for predictions
        loss = criterion(predictions, mini_batch_y.resize(mini_batch_y.size(0)*mini_batch_y.size(1)))


        LSTM.zero_grad()
        loss.backward()

        #gradient clipping might prove helpful    
        torch.nn.utils.clip_grad_norm_(LSTM.parameters(), 0.5) 
         
        optimizer.step()

        
        if iteration % generate_text == 0 or iteration == 1:
              
              method='MULTINOMIAL'
              
              start_state_h, start_state_c = nn_models.initialize_zero_state(device,1,lstm_size)
              
              #generate text using model
              
              generated_text = nn_models.generate_text(device,LSTM, start_state_h, 
                                                 start_state_c,
                                                 len(int_to_vocab),
                                                 vocab_to_int, int_to_vocab, 
                                                 starting_token="The",  
                                                 sample_words=8, lenght=200,
                                                 sampling=method)
              

              #save generated text
              if method == 'TOPK':
                    
                    save_path = "results/generated articles word_level_rnn_topk"                    

              else:
                    save_path = "results/generated articles word_level_rnn_multinomial"
                    
                    
              if 'tech' in file_path:
                   
                    torch.save(LSTM.state_dict(), 'models/word_level_rnn/bbc_model-tech-iteration{}.pth'.format(iteration))   
                    dataset='tech'
              else:
                    torch.save(LSTM.state_dict(), 'models/word_level_rnn/bbc_model-sports-iteration{}.pth'.format(iteration))   
                    dataset='sports'      
                    
              
              utils.save_generated_text(generated_text,save_path,iteration,dataset=dataset,method=method) 
                
              
              
        
        if iteration % save_loss == 0:
            training_losses.append(loss.item()) #categorical cross-entropy loss
            print ('Iteration [{}/{}], Loss: {:.3f}, '.format(iteration, epochs*num_of_batches, loss.item()),
                   'Epoch: [{}/{}] '.format(epoch,epochs))
            






end_train = time.time()
print('Completed Training with : {} iterations '.format(epochs*num_of_batches))        
print ("Training completed at {0:.2f} seconds".format(end_train-start_train))



##PLOT LOSS AND PERPLEXITY BELOW
model_name='Word-level RNN'




if "tech" in file_path:
      graph_path="results/performance graphs/Word_RNN_(Tech)"
            
      nn_models.plot_loss(training_losses,model_name,graph_path) #plot categorical cross-entropy loss
      perplexities=nn_models.plot_perplexity(training_losses,model_name,graph_path) #plot training perplexity
      #save metrics in directory
      with open('results/performance  saved mertics/word_rnn_loss_tech.pickle','wb') as f: pickle.dump(training_losses, f)
      with open('results/performance  saved mertics/word_rnn_perplexity_tech.pickle','wb') as f: pickle.dump(perplexities, f)


else:
      graph_path="results/performance graphs/Word_RNN_(Sports)"
      
      nn_models.plot_loss(training_losses,model_name,graph_path) #plot categorical cross-entropy loss
      perplexities=nn_models.plot_perplexity(training_losses,model_name,graph_path) #plot training perplexity
      
      #save metrics in directory
      with open('results/performance  saved mertics/word_rnn_loss_sports.pickle','wb') as f: pickle.dump(training_losses, f)
      with open('results/performance  saved mertics/word_rnn_perplexity_sports.pickle','wb') as f: pickle.dump(perplexities, f)


















