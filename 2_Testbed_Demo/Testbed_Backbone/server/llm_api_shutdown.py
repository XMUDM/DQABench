"""
Example call:
python llm_api_shutdown.py --serve all
The value can be "all","controller","model_worker", or "openai_api_server". all indicates that all services are stopped
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--serve", choices=["all", "controller", "model_worker", "openai_api_server"], default="all")

args = parser.parse_args()

base_shell = "ps -eo user,pid,cmd|grep fastchat.serve{}|grep -v grep|awk '{{print $2}}'|xargs kill -9"

if args.serve == "all":
    shell_script = base_shell.format("")
else:
    serve = f".{args.serve}"
    shell_script = base_shell.format(serve)

subprocess.run(shell_script, shell=True, check=True)
print(f"llm api sever --{args.serve} has been shutdown!")
