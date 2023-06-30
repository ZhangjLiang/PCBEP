import argparse
import os


def remove_more_mods(input, mpath, output):
    with open(input, 'r') as f:
        for line in f:
            row = line.strip().split(':')
            pdb = row[0]
            fw = open(output + str(pdb) + '.pdb', 'w')
            with open(mpath + str(pdb) + '.pdb', 'r') as fp:
                li = fp.readlines()
            for count in range(len(li)):
                if li[count].strip().split()[0] == 'MODEL' and li[count].strip().split()[1] == '2':
                    break
                fw.write(li[count].strip().split('\n')[0] + '\n')


def main(m_path, m_pdb_path, one_pdb_path):
    input = './data/more_mods.txt'
    mpath = './data/more_mods/'
    output = './data/one_pdb/'

    remove_more_mods(input, mpath, output)
    print("Done removing model!")


if __name__ == '__main__':
    main()
