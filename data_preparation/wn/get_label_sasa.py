import os
import uuid
from typing import Optional, Dict
from Bio.PDB import PDBParser, SASA
from Bio import PDB


class SurfaceArea:
    def __init__(self, probe_radius: float = 1.4, n_points: int = 100, radii_dict: Optional[Dict] = None):
        if radii_dict is None:
            # radii_dict = {'X': 2.0, 'LI': 1.82, 'NE': 1.54, 'SI': 2.1, 'CL': 1.75, 'AR': 1.88, 'CU': 1.4, 'ZN': 1.39,
            #               'GA': 1.87, 'AS': 1.85, 'SE': 1.9, 'BR': 1.85, 'KR': 2.02, 'AG': 1.72, 'CD': 1.58, 'IN': 1.93,
            #               'SN': 2.17, 'TE': 2.06, 'XE': 2.16, 'PT': 1.72, 'AU': 1.66, 'HG': 1.55, 'TL': 1.96, 'PB': 2.02,
            #               'U': 1.86, 'EU': 1.47}
            radii_dict = {'X': 2.0}

        self.parser = PDBParser(QUIET=1)
        self.structure_computer = SASA.ShrakeRupley(probe_radius=probe_radius, n_points=n_points, radii_dict=radii_dict)

    def __call__(self, name, loc, chain) -> float:
        struct = self.parser.get_structure(name, loc)
        # self.structure_computer.compute(struct, level="C")
        # return struct[0]['C'].sasa
        self.structure_computer.compute(struct, level="R")
        return struct[0][chain].child_list


class ChainSplitter:
    def __init__(self, out_dir=None):
        """ Create parsing and writing objects, specify output directory. """
        self.parser = PDB.PDBParser()
        self.writer = PDB.PDBIO()
        if out_dir is None:
            out_dir = os.path.join(os.getcwd(), "chain_PDBs")
        self.out_dir = out_dir

    def make_pdb(self, pdb_path, chain_letters, overwrite=False, struct=None):

        pdb_id = self.out_dir.split('\\')[-1]
        # Get structure, write new file with only given chains
        if struct is None:
            struct = self.parser.get_structure(pdb_id, self.out_dir)
        self.writer.set_structure(struct)
        self.writer.save(pdb_path, select=SelectChains(chain_letters))

        return pdb_path


class SelectChains(PDB.Select):
    """ Only accept the specified chains when saving. """

    def __init__(self, chain_letters):
        self.chain_letters = chain_letters

    def accept_chain(self, chain):
        return (chain.get_id() in self.chain_letters)


def get_sasa_label(complex_sasa, antigen_sasa):
    surface_list = ''
    for complex, antigen in zip(complex_sasa, antigen_sasa):
        s_value = float(antigen.sasa) - float(complex.sasa)
        if s_value > 1.0:
            surface_list += '1'
        else:
            surface_list += '0'
    return surface_list


def get_differenct_sasa(complex_sasa, antigen_sasa, num):
    surface_list = ''
    surface_value = []
    for complex, antigen in zip(complex_sasa, antigen_sasa):
        s_value = float(antigen.sasa) - float(complex.sasa)
        if s_value > 1.0:
            surface_list += '1'
        else:
            surface_list += '0'
    return surface_list


def get_surface(antigen_sasa):
    surface_list = ''
    for antigen in antigen_sasa:
        if float(antigen.sasa) > 1.0:
            surface_list += '1'
        else:
            surface_list += '0'
    return surface_list


# 调用函数生成label.txt文件
def get_sasa_difference(complex_name, antigen, antigen_file, complex_file):
    com_path = './data/pdb_sasa/'
    an_path = './data/one_pdb/'

    label = ''

    sasa_fn = SurfaceArea()
    complex_sasa = sasa_fn(uuid.uuid4().hex, com_path + complex_file, antigen)
    antigen_sasa = sasa_fn(uuid.uuid4().hex, an_path + antigen_file, antigen)
    surface_list = get_surface(antigen_sasa)
    label_sasa = get_sasa_label(complex_sasa, antigen_sasa)
    for i, j in zip(surface_list, label_sasa):
        if i == '1' and j == '1':
            label += '1'
        else:
            label += '0'

    return label


def main(input_list, label_path):
    if not os.path.exists(label_path):
        os.mkdir(label_path)
    with open(input_list, 'r') as f:
        with open(label_path + 'label.txt', 'w') as la:
            for line in f:
                name = line[:-1].replace('\t', '')  # 8EB2-BKL
                antigen = name[5]  # B
                antigen_file = name[:6] + '.pdb'
                complex_file = name[:6] + '.pdb'
                    # for l_line in li:
                    #     lines = l_line.strip().split()
                    #     name = lines[0]
                #     chain = lines[1]
                la.write('>' + name[:6] + '\n')
                la.write(get_sasa_difference(name.upper(), antigen, antigen_file, complex_file) + '\n')
                print(complex_file + 'over')


if __name__ == "__main__":
    main()
