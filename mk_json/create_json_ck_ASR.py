#!/usr/bin/env python

import json
import argparse
import os
import numpy as np

out_file = "t2_res_U0000000237.json"

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--result_file',  type = str)
	parser.add_argument('--mk_json',  type = str, help = "json directory")
	
	args = parser.parse_args()
	return args

def gt_code(out):
	if out == 0:
		code='020121'
	elif out == 1:
		code='02051'
	elif out == 2:
		code='020811'
	elif out == 3:
		code='020819'
	else:
		code='000001'
	return code
def rm_blank(new_l):
    last_l=[]
    for one in new_l:
        if one != '':
            last_l.append(one)
    return last_l
def mk_result():    

    with open(os.path.join(args.result_file), 'r') as NonResult :
        seg_info_tmp = NonResult.read().split("\n")

    seg_info = rm_blank(seg_info_tmp)

    result_dict={}
    result_dict['annotations']=[]
    for idx in range(len(seg_info)):		
        result = seg_info[idx].split()
        wav=result[0][:-4]+'.wav'
        target=int(result[1])
        class_code=gt_code(target)
        result_dict['annotations'].append({'file_name': '%s' %wav, 'class code': class_code})
        
    with open(os.path.join(args.mk_json+out_file), 'w') as json_file:
        json.dump(result_dict,json_file, ensure_ascii = False, indent = '    ', sort_keys = False)
		
if __name__ == "__main__":
	args = get_args()
	mk_result()
