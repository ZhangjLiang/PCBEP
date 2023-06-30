import os
import glob
import argparse
import shutil

'''
Search out pdb files with multiple mods
'''


def main(m_path, one_pdb_path, m_pdb_path):
    opath = m_path
    ipath = one_pdb_path + '*'
    npath =  m_pdb_path

    ipath = ipath
    for file_abs in glob.glob(ipath):
        # print(file_abs)
        # list.append(i)
        max = 0
        with open(file_abs, 'r') as f:
            for line in f:
                row = line.strip().split()
                if len(row) == 2 and row[0] == 'MODEL':
                    if max < int(row[1]):
                        max = int(row[1])
        if max != 0:
            file_path, file_name = os.path.split(file_abs)
            # os.rename(file_name,file_name[:-3]+ '-mods' + '.pdb')
            if not os.path.exists(npath):
                os.makedirs(npath)
            shutil.move(file_abs, npath + file_name)
            print("move %s -> %s" % (file_abs, npath + file_name))

            # print(file_abs[-10:-4] + ':' + str(max) + '\n')

            with open(opath, 'w') as fw:
                fw.write(file_abs[-10:-4] + ':' + str(max) + '\n')
    print("done generating more models!")


if __name__ == "__main__":
    main()
