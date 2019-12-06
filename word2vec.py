import torch
from random import shuffle
from collections import Counter
import argparse
import random
from huffman import HuffmanCoding
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def Analogical_Reasoning_Task(embedding, word2ind, ind2word,vocab):
    #######################  Input  #########################
    # embedding : Word embedding (type:torch.tesnor(V,D))   #
    #########################################################
    analogy_sample = open('questions-words.txt', encoding='utf-8').read()
    analogy_sample = analogy_sample.split('\n')[1:]
    analogy_sample = [i.split() for i in analogy_sample]
    analogy_sample = [i for i in analogy_sample if len(i) == 4]
    analogy = []
    for i in analogy_sample:
        if i[0] in vocab and i[1] in vocab and i[2] in vocab and i[3] in vocab:
            analogy.append(i)

    emb_vector = torch.stack([(embedding[word2ind[t[1]]] - embedding[word2ind[t[0]]] + embedding[word2ind[t[2]]]) for t in analogy])
    print(len(analogy))
    print(len(emb_vector))
    word_prediction=[]
    for i in emb_vector:
        length = (i * i).sum() ** 0.5
        inputVector = i.reshape(1, -1) / length
        sim = torch.mm(inputVector, embedding.t())[0] / length
        value, indice = sim.squeeze().topk(1)
        word_prediction.append(ind2word[int(indice)])
    word_analogy=[(j[0],j[1],j[2],t) for j,t in zip(analogy,word_prediction)]
    print(len(word_analogy))
    print(word_analogy[:100])




def subsampling(word_seq):
    ###############################  Output  #########################################subsample
    # subsampled : Subsampled sequence                                               #
    ##################################################################################

    coeff= 0.001
    subsample_list = []
    for i in range(len(word_seq.keys())):
        word_fraction = torch.Tensor([word_seq[i] / sum(word_seq.values())])
        subsample = (torch.sqrt(word_fraction/ coeff) + 1) * (coeff / word_fraction)
        subsample_list.append(subsample)
    return subsample_list


def skipgram_HS(centerWord, contextCode, inputMatrix, outputMatrix):
################################  Input  ########################################## 
# centerWord : Index of a centerword (type:int)                                   #
# contextCode : Code of a contextword (type:str)                                  #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Activated weight matrix of output (type:torch.tesnor(K,D))       #
###################################################################################
    hidden_layer=inputMatrix[centerWord]
    hidden_layer=[hidden_layer]
    hidden_layer=torch.stack(hidden_layer)
    output_layer=torch.mm(hidden_layer,outputMatrix.t())
    e = torch.exp(-output_layer)
    sigmoid=1/(1+e)
###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.t)
#                    #
###################################################################################
    loss=0
    codelist=torch.Tensor([int(i) for i in contextCode])
    loss_list=torch.stack([x-y for x,y in zip (sigmoid, codelist)])
    for i in loss_list:
        for j in i:
            if j<0:
                loss-=torch.log(j+1)
            else:
                loss-=torch.log(1-j)

    loss=loss/len(codelist)


    grad_out =torch.mm(loss_list.t(),hidden_layer)
    grad_in=torch.mm(loss_list, outputMatrix)

    return loss, grad_in, grad_out



def skipgram_NS(centerWord, inputMatrix, outputMatrix):
################################  Input  ##########################################
# centerWord : Index of a centerword (type:int)                                   #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Activated weight matrix of output (type:torch.tesnor(K,D))       #
###################################################################################
    hidden_layer = inputMatrix[centerWord]
    hidden_layer = [hidden_layer]
    hidden_layer = torch.stack(hidden_layer)
    output_layer = torch.mm(hidden_layer, outputMatrix.t())
    e=torch.exp(output_layer)
    softmax=e/e.sum()

###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(K,D))                    #
###################################################################################

    loss=0
    for i in softmax:
        loss-=torch.log(i[-1])

    dsoftmax=softmax.clone()
    for prob in dsoftmax:
        prob[-1]-=1

####################################################################################
    #loss=0 
    #codelist=torch.Tensor([0 for _ in range(19)] + [1 for _ in range(1)])
    #loss_list=torch.stack([x-y for x,y in zip (softmax, codelist)]) #start of backpropagation

    #for i in loss_list:
        #for j in i:
            #if j<0:
                #loss-=torch.log(j+1)
            #else:
                #loss-=torch.log(1-j)
#####################################################################################


    grad_out = torch.mm(dsoftmax.t(), hidden_layer)
    grad_in = torch.mm(dsoftmax, outputMatrix)


    return loss, grad_in, grad_out


def CBOW_HS(contextWords, centerCode, inputMatrix, outputMatrix):
################################  Input  ##########################################
# contextWords : Indices of contextwords (type:list(int))                          #
# centerCode : Code of a centerword (type:str)                                    #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Activated Weight matrix of output (type:torch.tesnor(K,D))       #
###################################################################################
    hidden_layer=0
    for i in contextWords:
        hidden_layer+=inputMatrix[i]

    hidden_layer = [hidden_layer]
    hidden_layer = torch.stack(hidden_layer)
    output_layer=torch.mm(hidden_layer,outputMatrix.t())
    e = torch.exp(-output_layer)
    sigmoid = 1/(1+e)
###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(K,D))                    #
###################################################################################
    loss=0
    codelist=torch.Tensor([float(i) for i in centerCode])
    loss_list=torch.stack([x-y for x,y in zip (sigmoid, codelist)]) #start of backpropagation

    for i in loss_list:
        for j in i:
            if j<0:
                loss-=torch.log(j+1)
            else:
                loss-=torch.log(1-j)



    grad_out =torch.mm(loss_list.t(),hidden_layer)
    grad_in=torch.mm(loss_list, outputMatrix)

    return loss, grad_in, grad_out


def CBOW_NS(contextWords, inputMatrix, outputMatrix):
################################  Input  ##########################################
# contextWords : Indices of contextwords (type:list(int))                         #
# inputMatrix : Weight matrix of input (type:torch.tesnor(V,D))                   #
# outputMatrix : Activated Weight matrix of output (type:torch.tesnor(K,D))       #
###################################################################################
    hidden_layer=0
    for i in contextWords:
        hidden_layer+=inputMatrix[i]
    hidden_layer = [hidden_layer]
    hidden_layer = torch.stack(hidden_layer)
    output_layer=torch.mm(hidden_layer,outputMatrix.t())
    e = torch.exp(output_layer)
    softmax = e/e.sum()
###############################  Output  ##########################################
# loss : Loss value (type:torch.tensor(1))                                        #
# grad_in : Gradient of inputMatrix (type:torch.tensor(1,D))                      #
# grad_out : Gradient of outputMatrix (type:torch.tesnor(K,D))                    #
###################################################################################
    loss=0
    for i in softmax:
        loss-=torch.log(i[-1])

    dsoftmax=softmax.clone()
    for prob in dsoftmax:
        prob[-1]-=1
	
####################################################################################
    #loss=0 
    #codelist=torch.Tensor([0 for _ in range(19)] + [1 for _ in range(1)])
    #loss_list=torch.stack([x-y for x,y in zip (softmax, codelist)]) #start of backpropagation

    #for i in loss_list:
        #for j in i:
            #if j<0:
                #loss-=torch.log(j+1)
            #else:
                #loss-=torch.log(1-j)
#####################################################################################

    grad_out = torch.mm(dsoftmax.t(), hidden_layer)
    grad_in = torch.mm(dsoftmax, outputMatrix)



    return loss, grad_in, grad_out


def word2vec_trainer(input_seq, target_seq, numwords, codes, stats, mode="CBOW", NS=20, dimension=100, learning_rate=0.025, epoch=3):
# train_seq : list(tuple(int, list(int))

# Xavier initialization of weight matrices
    W_in = torch.randn(numwords, dimension) / (dimension**0.5)
    W_out = torch.randn(numwords, dimension) / (dimension**0.5)
    i=0
    losses=[]
    print("# of training samples")
    print(len(input_seq))
    print()
    #stats = torch.LongTensor(stats)

    for _ in range(epoch):
        #Training word2vec using SGD(Batch size : 1)
        for inputs, output in zip(input_seq,target_seq):
            i+=1
            if mode=="CBOW":
                if NS==0:
                    #Only use the activated rows of the weight matrix
                    #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
                    activated=[]
                    current=codes[1]
                    for c in codes[0][output]:
                        activated.append(current.index)
                        if c=='0':
                            current=current.left
                        else:
                            current=current.right
                    L, G_in, G_out = CBOW_HS(inputs, codes[0][output], W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in
                    W_out[activated] -= learning_rate*G_out
                else:
                    #Only use the activated rows of the weight matrix
                    #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
                    shuffle(stats)
                    activated= [i for i in stats if i !=output][:NS]
                    activated.append(output)
                    L, G_in, G_out = CBOW_NS(inputs, W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in
                    W_out[activated] -= learning_rate*G_out

            elif mode=="SG":
                if NS==0:
                    #Only use the activated rows of the weight matrix
                    #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
                    activated=[]
                    current=codes[1]
                    for c in codes[0][output]:
                        activated.append(current.index)
                        if c=='0':
                            current=current.left
                        else:
                            current=current.right
                    L, G_in, G_out = skipgram_HS(inputs, codes[0][output], W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in.squeeze()
                    W_out[activated] -= learning_rate*G_out
                else:
                    #Only use the activated rows of the weight matrix
                    #activated should be torch.tensor(K,) so that activated W_out has the form of torch.tensor(K, D)
                    shuffle(stats)
                    activated =[i for i in stats if i != output][:NS]
                    activated.append(output)
                    L, G_in, G_out = skipgram_NS(inputs, W_in, W_out[activated])
                    W_in[inputs] -= learning_rate*G_in.squeeze()
                    W_out[activated] -= learning_rate*G_out



                
            else:
                print("Unkwnown mode : "+mode)
                exit()
            losses.append(L)
            if i % 1000==0:
            	avg_loss=sum(losses)/len(losses)
            	print("Loss {:d}: {:f}".format(i, avg_loss))
            	losses=[]

    return W_in, W_out


def main():
    parser = argparse.ArgumentParser(description='Word2vec')
    parser.add_argument('mode', metavar='mode', type=str,
                        help='"SG" for skipgram, "CBOW" for CBOW')
    parser.add_argument('ns', metavar='negative_samples', type=int,
                        help='0 for hierarchical softmax, the other numbers would be the number of negative samples')
    parser.add_argument('part', metavar='partition', type=str,
                        help='"part" if you want to train on a part of corpus, "full" if you want to train on full corpus')
    args = parser.parse_args()
    mode = args.mode
    part = args.part
    ns = args.ns

	#Load and preprocess corpus
    print("loading...")
    if part=="part":
        text = open('text8',mode='r').readlines()[0][:1000000] #Load a part of corpus for debugging
    elif part=="full":
        text = open('text8',mode='r').readlines()[0] #Load full corpus for submission
    else:
        print("Unknown argument : " + part)
        exit()

    print("preprocessing...")
    corpus = text.split()
    stats = Counter(corpus)
    words = []

    #Discard rare words
    for word in corpus:
        if stats[word]>4:
            words.append(word)
    vocab = set(words)

    #Give an index number to a word
    w2i = {}
    w2i[" "]=0
    i = 1
    for word in vocab:
        w2i[word] = i
        i+=1
    i2w = {}
    for k,v in w2i.items():
        i2w[v]=k


    #Code dict for hierarchical softmax
    freqdict={}
    freqdict[0]=10
    for word in vocab:
        freqdict[w2i[word]]=stats[word]
    codedict = HuffmanCoding().build(freqdict)




    #Frequency table for negative sampling
    freqtable = [0,0,0]
    for k,v in stats.items():
        f = int(v**0.75)
        for _ in range(f):
            if k in w2i.keys():
                freqtable.append(w2i[k])
    shuffle(freqtable)

    #Make training set
    print("build training set...")
    input_set = []
    target_set=[]
    window_size = 5

########################################################## subsampling code ##########################################################
# I just change this part of code as comment because it takes a long time. But according to the paper subsampling improve accuracy of analogy task
# Therefore if you want high accuracy I recommend you to use this part of code as well
# subsampling function is in the upper part of this function
    #subsample_class = []
    #for j in range(len(freqdict)):
        #subsample_classify = [1 for _ in range(int(subsampling(freqdict)[j] * 1000))] + [0 for _ in range(1000 - int(subsampling(freqdict)[j] * 1000))]
        #subsample_class.append(subsample_classify)
        #if len(subsample_class) % 50 == 0:
            #print(len(subsample_class))
########################################################################################################################################
    print('build training set2...')
    if mode=="CBOW":
        for j in range(len(words)):
            #choice=random.choice(subsample_class[w2i[words[j]]])
            #if choice==1:
                if j<window_size:
                    input_set.append([0 for _ in range(window_size-j)] + [w2i[words[k]] for k in range(j)] + [w2i[words[j+k+1]] for k in range(window_size)])
                    target_set.append(w2i[words[j]])
                elif j>=len(words)-window_size:
                    input_set.append([w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[len(words)-k-1]] for k in range(len(words)-j-1)] + [0 for _ in range(j+window_size-len(words)+1)])
                    target_set.append(w2i[words[j]])
                else:
                    input_set.append([w2i[words[j-k-1]] for k in range(window_size)] + [w2i[words[j+k+1]] for k in range(window_size)])
                    target_set.append(w2i[words[j]])
    if mode=="SG":
        for j in range(len(words)):
            if j < window_size:
                input_set += [w2i[words[j]] for _ in range(window_size * 2)]
                target_set += [0 for _ in range(window_size - j)] + [w2i[words[k]] for k in range(j)] + [
                w2i[words[j + k + 1]] for k in range(window_size)]
            elif j >= len(words) - window_size:
                input_set += [w2i[words[j]] for _ in range(window_size * 2)]
                target_set += [w2i[words[j - k - 1]] for k in range(window_size)] + [w2i[words[len(words) - k - 1]] for k in range(len(words) - j - 1)] + [0 for _ in range(j + window_size - len(words) + 1)]
            else:
                input_set += [w2i[words[j]] for _ in range(window_size * 2)]
                target_set += [w2i[words[j - k - 1]] for k in range(window_size)] + [w2i[words[j + k + 1]] for k in range(window_size)]

    print("Vocabulary size")
    print(len(w2i))
    print()

    #Training section
    emb,_ = word2vec_trainer(input_set, target_set, len(w2i), codedict, freqtable, mode=mode, NS=ns, dimension=64, epoch=1, learning_rate=0.01)
    Analogical_Reasoning_Task(emb,w2i,i2w,vocab)


main()
