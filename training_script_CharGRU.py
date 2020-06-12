"""
This script will be used to train a character-level GRU for generating text.

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

torch.manual_seed(0) #setting seed for reproduucible results
np.random.seed(0)

#processed data path, data file to load 
file_path="datasets/processed/bbc_tech.txt"

sequence_size=100
batch_size=64
training_text, output_text,num_of_batches, int_to_vocab, vocab_to_int = utils.load_processed_data_and_create_vocabulary_of_characters(file_path,
                                                                                                                        batch_size=batch_size,
                                                                                                                        sequence_size=sequence_size)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()

embedding_layer_size = 512 #
hidden_size=100  

#create GRU net
GRU = nn_models.GRU(vocabulary_size=len(int_to_vocab), embedding_dimensionality=embedding_layer_size, hidden_size=hidden_size)
GRU = GRU.to(device) 
GRU.train()
GRU.cuda()



optimizer = torch.optim.Adam(GRU.parameters(), lr=0.001)#,weight_decay=0.001 
iteration = 0
epochs= 500  
generate_text = 5_000 # generate text and save model after x iterations
save_loss = 100 #save loss after x iterations
training_losses=[]

start_train = time.time()

for epoch in range(1,epochs+1):

    state = nn_models.initialize_zero_state_gru(batch_size,hidden_size,device)
    
    batches = utils.generate_data_minibatch(training_text, output_text, 
                                            seq_size=sequence_size, 
                                            num_of_batches=num_of_batches)
    
    for mini_batch_x, mini_batch_y in batches:
        
        iteration+=1
          
          
        # Create tensors and put on the CUDA device
        mini_batch_x = torch.tensor(mini_batch_x).to(device, dtype=torch.long)
        mini_batch_y = torch.tensor(mini_batch_y).to(device, dtype=torch.long)   
                
        
        #detach states before feeding them to the network  

        predictions, state = GRU(mini_batch_x, state)
        
        state = state.detach()        
        
        loss = criterion(predictions, mini_batch_y.resize(mini_batch_y.size(0)*mini_batch_y.size(1)))

        GRU.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(GRU.parameters(), 0.5) 
         
        optimizer.step()
        
        
        
        if iteration % generate_text == 0 or iteration == 1: #save the first iteration and every x after that 
              
              method='MULTINOMIAL'
              
              initialized_state = nn_models.initialize_zero_state_gru(1,hidden_size,device)
              
              #generate text using model
              
              generated_text = nn_models.generate_text_with_gru_chars(device, GRU, initialized_state, len(int_to_vocab),
                                                 vocab_to_int, int_to_vocab, 
                                                 starting_token='A',
                                                 sample_tokens=6, lenght=350,
                                                 sampling=method)
              
                              
              #save generated text
              if method == 'TOPK':
                    save_path = "results/generated articles character_gru_topk"                    
              else:
                    save_path = "results/generated articles character_gru_multinomial"

              if 'tech' in file_path:
                    torch.save(GRU.state_dict(), 'models/character_level_GRU/bbc_model-tech-iteration{}.pth'.format(iteration))   
                    dataset='tech'
              else:
                    torch.save(GRU.state_dict(), 'models/character_level_GRU/bbc_model-sports-iteration{}.pth'.format(iteration))   
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
model_name='Character-level GRU'

if "tech" in file_path:      
      graph_path="results/performance graphs/CHAR_GRU_(Tech)"

      nn_models.plot_loss(training_losses,model_name,graph_path) #plot categorical cross-entropy loss
      perplexities=nn_models.plot_perplexity(training_losses,model_name,graph_path) #plot training perplexity

      
      with open('results/performance  saved mertics/char_gru_loss_tech.pickle','wb') as f: pickle.dump(training_losses, f)
      with open('results/performance  saved mertics/char_gru_perplexity_tech.pickle','wb') as f: pickle.dump(perplexities, f)


else:
      graph_path="results/performance graphs/CHAR_GRU_(Sports)"

      nn_models.plot_loss(training_losses,model_name,graph_path) #plot categorical cross-entropy loss
      perplexities=nn_models.plot_perplexity(training_losses,model_name,graph_path) #plot training perplexity
      
      with open('results/performance  saved mertics/char_gru_loss_sports.pickle','wb') as f: pickle.dump(training_losses, f)
      with open('results/performance  saved mertics/char_gru_perplexity_sports.pickle','wb') as f: pickle.dump(perplexities, f)

      











