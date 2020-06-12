"""
NN models are defined here to be used in the training  script
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0) #setting seed for reproduucible results
np.random.seed(0)


#####
####   First one is an LSTM
####

class LSTM(nn.Module):

    def __init__(self, vocabulary_size, embedding_dimensionality=256, lstm_size=256):
        super(LSTM, self).__init__()

        self.lstm_size=lstm_size
        
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_dimensionality)

        self.lstm_cell = nn.LSTM(embedding_dimensionality, lstm_size, num_layers=1 ,batch_first=True )
        
        self.fully_connected_layer = nn.Linear(lstm_size, vocabulary_size)


    def forward(self, input_x, previous_states):
        #previous states are passed as arguments
        #if no previous states exist then only zero vectors are passed

        embedding_output = self.embedding_layer(input_x)
        
        lstm_output, hidden_states = self.lstm_cell(embedding_output, previous_states)

        # Reshape output 
        lstm_output = lstm_output.reshape(lstm_output.size(0)*lstm_output.size(1), lstm_output.size(2))
        
        logit_output = self.fully_connected_layer(lstm_output)

        return logit_output, hidden_states




class GRU(nn.Module):
    def __init__(self, vocabulary_size,embedding_dimensionality=256, hidden_size=128):
        super(GRU, self).__init__()
        
        self.hidden_size = hidden_size

        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_dimensionality)
        
        self.gru_cell = nn.GRU(embedding_dimensionality, hidden_size, batch_first=True)
        
        self.fully_connected_layer = nn.Linear(hidden_size, vocabulary_size)


    def forward(self, input, hidden):

        embedding_output = self.embedding_layer(input)
        
        gru_output, hidden = self.gru_cell(embedding_output, hidden)

        # Reshape output 
        gru_output = gru_output.reshape(gru_output.size(0)*gru_output.size(1), gru_output.size(2))
        
        logit_output = self.fully_connected_layer(gru_output)        
        
        return logit_output, hidden




class Stacked_GRU(nn.Module):

    def __init__(self, vocabulary_size, embedding_dimensionality=256, lstm_size=256):
        super(Stacked_GRU, self).__init__()

        self.lstm_size=lstm_size
        
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_dimensionality)

        self.lstm_cell = nn.LSTM(embedding_dimensionality, lstm_size, num_layers=3 ,batch_first=True )
        
        self.fully_connected_layer = nn.Linear(lstm_size, vocabulary_size)


    def forward(self, input_x, previous_states):
        #previous states are passed as arguments
        #if no previous states exist then only zero vectors are passed

        embedding_output = self.embedding_layer(input_x)
        
        lstm_output, hidden_states = self.lstm_cell(embedding_output, previous_states)

        # Reshape output 
        lstm_output = lstm_output.reshape(lstm_output.size(0)*lstm_output.size(1), lstm_output.size(2))
        
        logit_output = self.fully_connected_layer(lstm_output)

        return logit_output, hidden_states



#reset, lstm states at the beginning of each epoch
#should also reset hidden states at inference time

def initialize_zero_state(device,batch_size, lstm_size): 
    #initialize hidden and cell states as 0s
    x=torch.zeros(1, batch_size, lstm_size).to(device)
    y=torch.zeros(1, batch_size, lstm_size).to(device)

    return(x,y) 








def generate_text(device, rnn, start_state_h, start_state_c, n_vocab, vocab_to_int, int_to_vocab, starting_token='Technology' ,sample_words=8,lenght=200,sampling="TOPK"):
    
    
    #use this function along with another sub-function.
    # input only one word for generating the text.
    #network is used for inference 
    rnn.eval()
    hidden_states=(start_state_h,start_state_c)

    generated_text=""+starting_token #or generated_text=None   Text to return as generated from the RNN
    

    starting_token_index = [[vocab_to_int[starting_token]]]

    #print(starting_token_index)

    starting_token_tensor= torch.tensor(starting_token_index).to(device, dtype=torch.long)

    #print(starting_token_tensor.size())
    
    with torch.no_grad():


        for _ in range(1,lenght+1):

            output, hidden_states = rnn(starting_token_tensor,hidden_states) # pass the index through the network
        
            if sampling == "MULTINOMIAL": 
                probabilities=output.exp()
                new_token_index = probabilities.multinomial(1).tolist()
                
                next_token_index = new_token_index[0][0]


            elif sampling == "TOPK":

                top_tokens = output.topk(sample_words)  #sample the x most probable words
                top_token_indices = top_tokens[1].tolist()  # [1] returns word indices, [0] returns the actual probability

                next_token_index = np.random.choice(top_token_indices[0]) # pick a random token out of the most likely tokens
                

            chosen_token = int_to_vocab[next_token_index]
            
            if sampling == "TOPK":

                generated_text = generated_text+" "+chosen_token   # adding space between the tokens here
            
            elif sampling == "MULTINOMIAL":

                generated_text = generated_text+""+chosen_token   



            starting_token_tensor=torch.tensor([[next_token_index]]).to(device, dtype=torch.long)


    print("GENERATED TEXT snippet : \n") 
    print(generated_text)

    rnn.train()   # return the network in training format 

    return  generated_text



def generate_text_with_chars(device, rnn, start_state_h, start_state_c, vocab_to_int, int_to_vocab ,sample_tokens=5,lenght=200,sampling="TOPK"):
    
    
    rnn.eval()
    hidden_states=(start_state_h,start_state_c)
    
    # Starting the sentance with a capitalized character.
    #starting token to be chosen at random from the vocabulary
    starting_token=np.random.choice(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U'])
    
    #Find a random starting capitalized character that is in the dictionary 
    while starting_token not in vocab_to_int:
            starting_token=np.random.choice(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U'])

    #starting_token='A'
    starting_token_index = [[vocab_to_int[starting_token]]] #find the integer representation of the character
    generated_text=""+starting_token #start sentence with the random character
    starting_token_tensor= torch.tensor(starting_token_index).to(device, dtype=torch.long)
    
    with torch.no_grad():


        for _ in range(1,lenght+1):

            output, hidden_states = rnn(starting_token_tensor,hidden_states) 

            if sampling == "MULTINOMIAL": 
                probabilities=output.exp()                
                new_token_index = probabilities.multinomial(1).tolist()
                next_token_index = new_token_index[0][0]



            elif sampling == "TOPK":


                top_tokens = output.topk(sample_tokens)  
                top_token_indices = top_tokens[1].tolist()  
                next_token_index = np.random.choice(top_token_indices[0]) 
                

            chosen_token = int_to_vocab[next_token_index]
            #generated text does not need a whitespace 
            generated_text = generated_text+""+chosen_token   

            starting_token_tensor=torch.tensor([[next_token_index]]).to(device, dtype=torch.long)


    print("GENERATED TEXT snippet : \n")
    print(generated_text) 

    rnn.train()   

    return  generated_text




def detach(states):
    return [state.detach() for state in states]






def generate_text_with_gru_chars(device, rnn, start_state_h, n_vocab, vocab_to_int, int_to_vocab, starting_token='Technology' ,sample_tokens=8,lenght=200,sampling="TOPK"):
    

    rnn.eval()


    starting_token=np.random.choice(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U'])    
    #Find a random starting capitalized character that is in the dictionary 
    while starting_token not in vocab_to_int:
            starting_token=np.random.choice(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U'])

    generated_text=""+starting_token 
    starting_token_index = [[vocab_to_int[starting_token]]]
    starting_token_tensor= torch.tensor(starting_token_index).to(device, dtype=torch.long)

    hidden_state = start_state_h
    with torch.no_grad():


        for _ in range(1,lenght+1):

            output, hidden_state = rnn(starting_token_tensor,hidden_state) # pass the index through the network

            if sampling == "MULTINOMIAL": 

                probabilities=output.exp()
                new_token_index = probabilities.multinomial(1).tolist()
                next_token_index = new_token_index[0][0]



            elif sampling == "TOPK":

                top_tokens = output.topk(sample_words)  
                top_token_indices = top_tokens[1].tolist()  
                next_token_index = np.random.choice(top_token_indices[0]) 

            
            chosen_token = int_to_vocab[next_token_index]
            generated_text = generated_text+""+chosen_token   

            starting_token_tensor=torch.tensor([[next_token_index]]).to(device, dtype=torch.long)


    print("GENERATED TEXT snippet : \n") 
    print(generated_text)

    rnn.train()   # return the network in training format 

    return  generated_text





def initialize_zero_state_gru(batch_size, hidden_size, device):
    return torch.zeros(1, batch_size, hidden_size, device=device)





def plot_loss(loss,model,save_path):
    #loss -> list
    #plot the training loss and save in the given directory
    plt.plot(loss)
    plt.title('Training loss: '+str(model))
    plt.xlabel('Training iterations', fontsize=11)
    plt.ylabel('Cross-entropy Loss', fontsize=11)
    plt.savefig(save_path+'_training_loss.png')  
    plt.show()




def plot_perplexity(loss,model,save_path):

    # Calculate the perplexity using torch.
    perplexity=[]
    for loss_item in loss:
        perplexity.append(torch.tensor([loss_item],dtype=torch.float32).exp())

    plt.plot(perplexity)
    plt.title('Training perplexity: '+str(model))
    plt.xlabel('Training iterations', fontsize=11)
    plt.ylabel('Perplexity', fontsize=11)
    plt.savefig(save_path+'_training_perplexity.png')
    plt.show()  
    return perplexity














































