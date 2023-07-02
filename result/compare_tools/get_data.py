import pandas as pd
def get_label():
    lable_list = []
    with open('40_label.txt') as fl:
        for line in fl:
            label = fl.readline().strip()
            if len(label) > 1023:
                print(line.strip())
            for i in label:
                lable_list.append(i)
    return lable_list


def get_score(path):
    score_list = []
    # with open(path) as fp:
    #     for line in fp:
    #         name = line.strip().split()[0]
    #         score = line.strip().split()[2]
    #         if name not in ['7UOT-A','7YRF-C','8FR6-C','8BBO-H','7PC2-D','7TN0-E']:
    #             score_list.append(score)
    with open(path) as fp:
        for line in fp:
            score = line.strip().split()[0]
            score_list.append(score)
    return score_list


def write_data(path, label_list, score_list):
    with open(path, 'w') as fp:
        for i in range(len(label_list)):
            fp.write(str(label_list[i]) + '\t' + score_list[i] + '\n')

if __name__ == '__main__':
    label_list = get_label()
    score_list = get_score('sema_score.txt')
    write_data('sema1D.txt', label_list, score_list)
