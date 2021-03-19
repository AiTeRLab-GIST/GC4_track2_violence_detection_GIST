#!/usr/bin/env python
# coding: utf-8

import os
import sys

if __name__=='__main__':

    folder_name = sys.argv[1]
    
    os.system('./phj/run_4th_iitp.sh '+str('./txt/google_ts/')+' ' + str('./result/'))
    os.system('./mk_json/run.sh '+str('./result/phj_result.txt')+' '+str('/aichallenge/'))