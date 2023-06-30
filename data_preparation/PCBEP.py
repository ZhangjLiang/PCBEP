import get_pdb_from_website as down_csv,generate, get_pssm
import select_data, get_one_ag_data, down_pdb, cdhit_fasta, get_ag_at
from wn import update_pdb, ag_at, antigen_cat, divide_one_chain, get_fasta, get_more_mods, remove_mods, \
    divide_more_chain, get_label_sasa, get_surface
import requests
import xlrd
import pandas as pd
import os
from bs4 import BeautifulSoup


def process():

    # down data
    # down_csv.main('D:\work\chromedriver.exe', 'new_data.txt', '1.csv')
    # use your own chromedriver and path
    down_csv.main('E:/Download/chromedriver_win32/chromedriver.exe', './1.txt', '1.csv')
    # time must be 'year-month'
    # select data
    select_data.main('1.csv', 'data/data.xls', seq_len=50, after_time='1999-1', befor_time='2023-12')


def get_list_fasta_surface_label():
    #
    # ***get the pdb with one antigen  data.xls is original file from website
    #                               initial_list.txt is the output file
    #get_one_ag_data.get_data('data/data.xls', 'data/initial_list.txt')

    # ***download pdb from initial list
    #down_pdb.main(list_path= 'data/initial_list.txt',file_path= 'data/pdb68/')

    # *** remove HETATM in PDB
    # *** list original_path new_path
    #update_pdb.main('data/initial_list.txt', 'data/pdb68/', 'data/pdb_update/')

    # *** get antigen chain pdb eg
    # *** input list:data/initial_list.txt
    #divide_one_chain.divide('data/initial_list.txt', 'data/pdb_update/', 'data/one_pdb/')

    # *** get fasta from pdb
    #get_fasta.main('data/initial_list.txt', 'data/fasta/', 'data/one_pdb/')

    # ***
    #cdhit_fasta.main('/home/yan/cdhit/cdhit-4.6.2', '/home/yan/bcell/PCBEP_all_code/data/fasta/echo_fasta.txt',
    #       '/home/yan/bcell/PCBEP_all_code/data/fasta.txt', 'data/no_repeat_list.txt')  # get list

    # *** get antigen and antibody chain name in fasta
    # eg.    1IFH	P	,	L	|	H
    #       pdbname     antigen | antibody
    #get_ag_at.main('data/data.xls', 'data/ag_at.txt','./data/fasta/fasta_total.txt')

    # update input new list(cd_hit result) : ag_at.txt
    #
    #get_more_mods.main('data/more_mods.txt', 'data/one_pdb/', 'data/more_mods/')

    #
    #remove_mods.main('data/more_mods.txt', 'data/more_mods/', 'data/one_pdb/')

    # *** input format 1IFH	P	,	L	|	H
    # *** output format 1IFH-P L H
    #ag_at.main('data/ag_at.txt', 'data/ag_at_inter.txt')

    #
    antigen_cat.main('data/ag_at_inter.txt','data/ag_at_inter_cat.txt')

    #
    #divide_more_chain.main('data/ag_at_inter.txt', 'data/pdb_update/', 'data/pdb_sasa/')

    #   surface
    #get_surface.main('data/one_pdb/', 'data/surface/', 'data/ag_at.txt')

    #   label
    #get_label_sasa.main('data/ag_at_inter.txt', 'data/label/')

    #   feature
    #get_pssm.main('/home/yan/bcell/PCBEP_all_code/data/fasta/', '/home/yan/bcell/PCBEP/PSSM/ncbi-blast-2.13.0+/bin', '/home/yan/bcell/PCBEP_all_code/data/pssm/' )

    # https://possum.erc.monash.edu/server.jsp
    #
    #generate.main()


if __name__ == '__main__':
    #process()
    get_list_fasta_surface_label()
