import argparse
import os


def select_fasta(old_fa, new_fa):
    with open(old_fa, 'r',) as of:
        with open(new_fa, 'w') as nf:
            for line in of:
                next_line = of.readline().strip()
                if len(next_line) > 50:
                    nf.write(line)
                    nf.write(next_line + '\n')

def main(data_list, fa_path, one_pdb_path):
    input = data_list
    fasta = fa_path
    model_one = one_pdb_path

    if not os.path.exists(fasta):
        os.mkdir(fasta)
    with open(fasta + 'fasta_total.txt', 'w') as fw:
    # fw = open(fasta + 'fasta.txt', 'w')
    # fw = open('./data/ind264_data/ind264_data_list.txt','w')

        with open(input, 'r') as fp:
            for lines in fp:
                row = lines.strip().split()
                pdb = row[0][:4].upper()
                chain = row[1]
                fw.write('>' + pdb.upper() + '-' + chain + '\n')

                ff = open(fasta + str(pdb) + '-' + str(chain) + '.fa', 'w')
                ff.write('>' + pdb.upper() + '-' + chain + '\n')

                with open(model_one + '{}-{}.pdb'.format(pdb, chain), 'r') as f:
                    m = -10000
                    for line1 in f:
                        row = line1.strip().split()
                        if (row[0] == 'ATOM' and row[4][0] == chain):
                            if (len(row[2]) == 7):
                                row[7] = row[6]
                                row[6] = row[5]
                                row[5] = row[4]
                                row[4] = row[3]
                                row[3] = row[2][4:]
                                row[2] = row[2][:4]
                            if (len(row[2]) == 8):
                                row[8] = row[7]
                                row[7] = row[6]
                                row[6] = row[5]
                                row[5] = row[4]
                                row[4] = row[3]
                                row[3] = row[2][4:]
                                row[2] = row[2][:4]
                            if (len(row[4]) != 1):
                                row.append('0')
                                # row[11] = row[10]
                                row[7] = row[6]
                                row[6] = row[5]
                                row[5] = row[4][1:]
                                row[4] = row[4][0]
                            if (len(row[3]) != 3):
                                row[3] = row[3][-3:]

                            if str(m) != row[5]:  # 氨基酸改变
                                # print(row[5])
                                # print(m)
                                # print(row[3])
                                if row[3] == 'GLY':
                                    fw.write('G')
                                    ff.write('G')
                                elif row[3] == 'ALA':
                                    fw.write('A')
                                    ff.write('A')
                                elif row[3] == 'VAL':
                                    fw.write('V')
                                    ff.write('V')
                                elif row[3] == 'LEU':
                                    fw.write('L')
                                    ff.write('L')
                                elif row[3] == 'ILE':
                                    fw.write('I')
                                    ff.write('I')
                                elif row[3] == 'PRO':
                                    fw.write('P')
                                    ff.write('P')
                                elif row[3] == 'PHE':
                                    fw.write('F')
                                    ff.write('F')
                                elif row[3] == 'TYR':
                                    fw.write('Y')
                                    ff.write('Y')
                                elif row[3] == 'TRP':
                                    fw.write('W')
                                    ff.write('W')
                                elif row[3] == 'SER':
                                    fw.write('S')
                                    ff.write('S')
                                elif row[3] == 'THR':
                                    fw.write('T')
                                    ff.write('T')
                                elif row[3] == 'CYS':
                                    fw.write('C')
                                    ff.write('C')
                                elif row[3] == 'MET':
                                    fw.write('M')
                                    ff.write('M')
                                elif row[3] == 'ASN':
                                    fw.write('N')
                                    ff.write('N')
                                elif row[3] == 'GLN':
                                    fw.write('Q')
                                    ff.write('Q')
                                elif row[3] == 'ASP':
                                    fw.write('D')
                                    ff.write('D')
                                elif row[3] == 'GLU':
                                    fw.write('E')
                                    ff.write('E')
                                elif row[3] == 'LYS':
                                    fw.write('K')
                                    ff.write('K')
                                elif row[3] == 'ARG':
                                    fw.write('R')
                                    ff.write('R')
                                elif row[3] == 'HIS':
                                    fw.write('H')
                                    ff.write('H')
                                m = str(row[5])
                fw.write('\n')
                ff.write('\n')

    select_fasta(fasta + 'fasta_total.txt', fasta + 'echo_fasta.txt')
    print("Done extracting fasta!")


if __name__ == "__main__":
    main()
