import os


def get_pssm(name, fasta_path, blast_path, pssm_path):
    os.chdir(blast_path)
    # os.chdir(path)
    os.makedirs(pssm_path, exist_ok=True)
    os.system(blast_path + '/psiblast -db swissprot '
              '-query ' + fasta_path + name.upper() + '.fa -evalue 0.001 -num_iterations 3 '
                '-out_ascii_pssm ' + pssm_path + name.upper() + '.pssm')
    with open(pssm_path + 'loss_pssm.txt', 'a') as fw:
        if not os.path.exists(pssm_path + name.upper() + '.pssm'):
            with open(fasta_path + name.upper() + '.fa') as f:
                fw.write(f.readline())


def main(fasta_path, blast_path, pssm_path):
    os.makedirs(fasta_path, exist_ok=True)
    import time

    with open(pssm_path + 'loss_pssm.txt', 'a') as fw:
        fw.write(time.ctime() + '\n')
    with open('/home/yan/bcell/PCBEP_all_code/data/fasta/fasta_total.txt') as fa:
        for line in fa:
            name = line.strip()[1:]
            fa.readline()
            get_pssm(name, fasta_path, blast_path, pssm_path)


if __name__ == '__main__':
    # get_pssm('/home/yan/bcell/static/data/user_1667820021/')
    # get_single_fasta()
    with open('64_indep_fasta.txt') as fa:
        for line in fa:
            name = line.strip()[1:]
            fa.readline()
            get_pssm(name)
