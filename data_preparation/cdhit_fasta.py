import os


def cdhit(cdhit_path, input_path, output_path):
    # os.chdir(cdhit_path)
    # os.chdir(path)
    os.system(cdhit_path + '/cd-hit -i ' + input_path +
              ' -o ' + output_path + ' -c 0.7 -n 3')


def get_new_list(clstr_path, out_path):
    with open(clstr_path) as nf:
        with open(out_path, 'w') as fw:
            for line in nf:
                if line.startswith('>'):
                    lines = line[1:].strip().split('-')
                    fw.write(lines[0] + '\t' + lines[1] + '\n')


def cdhit_2d(cdhit_path, input_path1, input_path2, output_path):
    os.system(cdhit_path + '/cd-hit-2d -i ' + input_path1 +
              ' -i2 ' + input_path2 + ' -o ' + output_path + ' -c 0.7 -n 3')


def main(cdhit_path, fa_path, out_fa_path, list_path):
    cdhit(cdhit_path, fa_path, out_fa_path)
    # get_new_list(out_fa_path, list_path)


if __name__ == '__main__':
    main('/home/yan/cdhit/cdhit-4.6.2', '/home/yan/bcell/PCBEP_all_code/data/fasta/echo_fasta.txt',
         '/home/yan/bcell/PCBEP_all_code/data/fasta.txt', 'data/no_repeat_list.txt')
