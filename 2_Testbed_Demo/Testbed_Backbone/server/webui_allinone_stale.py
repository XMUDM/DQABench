"""Usage
Load the local model:
python webui_allinone.py

Invoke the remote api service:
python webui_allinone.py --use-remote-api

Background operation webui service:
python webui_allinone.py --nohup

Load multiple non-default models:
python webui_allinone.py --model-path-address model1@host1@port1 model2@host2@port2 

Multiple gpus start:
python webui_alline.py --model-path-address model@host@port --num-gpus 2 --gpus 0,1 --max-gpu-memory 10GiB

"""
import streamlit as st
from webui_pages.utils import *
from streamlit_option_menu import option_menu
from webui_pages import *
import os
from server.llm_api_stale import string_args,launch_all,controller_args,worker_args,server_args,LOG_PATH

from server.api_allinone_stale import parser, api_args
import subprocess

parser.add_argument("--use-remote-api",action="store_true")
parser.add_argument("--nohup",action="store_true")
parser.add_argument("--server.port",type=int,default=8501)
parser.add_argument("--theme.base",type=str,default='"light"')
parser.add_argument("--theme.primaryColor",type=str,default='"#165dff"')
parser.add_argument("--theme.secondaryBackgroundColor",type=str,default='"#f5f5f5"')
parser.add_argument("--theme.textColor",type=str,default='"#000000"')
web_args = ["server.port","theme.base","theme.primaryColor","theme.secondaryBackgroundColor","theme.textColor"]


def launch_api(args,args_list=api_args,log_name=None):
    print("Launching api ...")
    if not log_name:
        log_name = f"{LOG_PATH}api_{args.api_host}_{args.api_port}"
    print(f"logs on api are written in {log_name}")
    args_str = string_args(args,args_list)
    api_sh = "python  server/{script} {args_str} >{log_name}.log 2>&1 &".format(
        script="api.py",args_str=args_str,log_name=log_name)
    subprocess.run(api_sh, shell=True, check=True)
    print("launch api done!")

def launch_webui(args,args_list=web_args,log_name=None):
    print("Launching webui...")
    if not log_name:
        log_name = f"{LOG_PATH}webui"

    args_str = string_args(args,args_list)
    if args.nohup:
        print(f"logs on api are written in {log_name}")
        webui_sh = "streamlit run webui.py {args_str} >{log_name}.log 2>&1 &".format(
        args_str=args_str,log_name=log_name)
    else:
        webui_sh = "streamlit run webui.py {args_str}".format(
        args_str=args_str)
    subprocess.run(webui_sh, shell=True, check=True)
    print("launch webui done!")


if __name__ == "__main__":
    print("Starting webui_allineone.py, it would take a while, please be patient. If it does not start for a long time, go to {LOG_PATH} to view logs....")
    args = parser.parse_args()

    print("*"*80)
    if not args.use_remote_api:
        launch_all(args=args,controller_args=controller_args,worker_args=worker_args,server_args=server_args)
    launch_api(args=args,args_list=api_args)
    launch_webui(args=args,args_list=web_args)
    print("Start webui_allinone.py done!")