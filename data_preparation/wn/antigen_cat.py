def main(input_path, output_path):
    fw = open(output_path, 'w')
    pre_name = '1'
    pre_at = '1'
    with open(input_path, 'r') as f:
        for line in f:
            row = line.strip().split('-')
            name = row[0]
            antigen = row[1][0]
            antibody = row[1][1:]
            # print()
            if name == pre_name and antibody == pre_at:
                continue
            pre_name = str(name)
            pre_at = str(antibody)
            fw.write(str(name) + '-' + str(antigen))
            flag = 0
            with open(input_path, 'r') as ff:
                for lii in ff:
                    r = lii.strip().split('-')
                    n = r[0]
                    ag = r[1][0]
                    ab = r[1][1:]
                    if name == n and antigen != ag and antibody == ab:
                        fw.write('-' + str(ag))
                    # else:
                    #     if antibody != ab:
                    #         flag =1
            if flag == 0:
                fw.write(' ' + str(antibody) + '\n')


if __name__ == "__main__":
    main('./data/new_ag_at.txt', './data/new_indep_list.txt')
