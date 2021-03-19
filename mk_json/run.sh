#!/bin/bash

result_file=$1
mk_json=$2

#python ./mk_json/mk_result.py
python ./mk_json/create_json_ck_ASR.py --result_file=$result_file --mk_json=$mk_json