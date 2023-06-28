import os
import torch
import esm
#import umap.umap_ as umap
import numpy as np
# set seqnum
seqnum = 1
# Load ESM-2 model

'''model = torch.load(
            f"./models/esm2_t33_650M_UR50D.pt",
        )
alphabet = torch.load(
            f"./models/esm2_t33_650M_UR50D-contact-regression.pt",
        )
model, alphabet = esm.pretrained.load_model_and_alphabet_core('esm2_t33_650M_UR50Desm2_t33_650M_UR50D', model, alphabet)
batch_converter = alphabet.get_batch_converter()
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")'''

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def write_embedding(num):
    #write 59
    fr = open('./'+str(num)+'_fasta.txt','r')
    fastalist = fr.readlines()
    count = 0
    for i in range(len(fastalist)-1):
        if fastalist[i][0] == '>':
            data = [
                (''+fastalist[i]+'',''+fastalist[i+1].upper()+'')
            ]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            fw = open('./esm_result/'+str(num)+'_'+str(count)+'.txt', 'w')
            # Extract per-residue representations (on CPU)
            batch_tokens = batch_tokens.to(device)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]
            # token_representations   seq num*seq len+2*1280
            list = token_representations.tolist()
            new_feature = list[0]
            for j in range(1,len(new_feature)-1):
                for k in range(len(new_feature[j])):
                    fw.write(str(new_feature[j][k]))
                    fw.write('\t')
                fw.write('\n')
            count +=1
            print(count)

#dataset num
write_embedding(724)


