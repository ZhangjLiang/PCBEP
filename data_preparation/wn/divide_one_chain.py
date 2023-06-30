from Bio.PDB import Select, PDBIO
from Bio.PDB.PDBParser import PDBParser
import os


class ChainSelect(Select):
    def __init__(self, chain):
        self.chain = chain

    def accept_chain(self, chain):
        # num = len(self.chain)
        if chain.get_id() in self.chain:
            return 1
        else:
            return 0


def divide(data_list, new_pdb_path, one_pdb_path):
    a = ''
    chains = list(a)
    if not os.path.exists(one_pdb_path):
        os.mkdir(one_pdb_path)
    with open(data_list, 'r') as fp:
        for line in fp:
            line = line.strip().split()
            name = line[0]
            chains = line[1]
            # for i in range(1,len(line[0].strip().split('-'))):
            #     chains = []
            #     chains.append(line[0].strip().split('-')[i])
            #     for i in range(1, len(line)):
            #         chains.append(line[i])
            # chains = line[1].strip().split()
            # chains = ['A','G','H','I','J','K','L']
            p = PDBParser(PERMISSIVE=1)
            pdb_file = new_pdb_path + '{}.pdb'.format(name)
            structure = p.get_structure(pdb_file, pdb_file)

            # for chain in chains:
            pdb_chain_file = one_pdb_path + '{}-{}.pdb'.format(name, ''.join(chains))
            io_w_no_h = PDBIO()
            io_w_no_h.set_structure(structure)
            io_w_no_h.save('{}'.format(pdb_chain_file), ChainSelect(chains))
            print('has bean down {}'.format(pdb_chain_file))
