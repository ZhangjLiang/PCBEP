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


def main(list_path, new_pdb_path, complex_path):
    if not os.path.exists(complex_path):
        os.mkdir(complex_path)
    a = ''
    chains = list(a)
    with open(list_path, 'r') as fp:
        for line in fp:
            line = line.strip().split()
            name = line[0][:4]

            for i in range(1, len(line[0].strip().split('-'))):
                chains = []
                chains.append(line[0].strip().split('-')[i])
                for i in range(1, len(line)):
                    chains.append(line[i])
                # chains = line[1].strip().split()
                # chains = ['A','G','H','I','J','K','L']
                p = PDBParser(PERMISSIVE=1)
                pdb_file = new_pdb_path + '{}.pdb'.format(name)
                structure = p.get_structure(pdb_file, pdb_file)

                # for chain in chains:
                pdb_chain_file = complex_path + '{}-{}.pdb'.format(name, ''.join(chains[0]))
                io_w_no_h = PDBIO()
                io_w_no_h.set_structure(structure)
                io_w_no_h.save('{}'.format(pdb_chain_file), ChainSelect(chains))
                print(pdb_chain_file)

    # print("Done extracting fasta!")


if __name__ == "__main__":
    main()
