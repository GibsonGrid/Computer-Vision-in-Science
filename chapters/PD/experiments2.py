import csv
import pandas as pd
import os 
import datetime
import pathlib
# import common
import platform
# df = pd.read_csv('experiments2.csv', index_col=False,sep=",")
df = pd.read_excel('experiments2_paper3.xlsx', engine='openpyxl')



#-----------------------------------------------
df1 = df[df["yes"]!=0]



cols = df1.columns
for i in range(df1.shape[0]):
    command = ""
    stamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    output_name = ""
    for c in cols:
        if c != "file" and c != "yes" :
            command += f" --{c} {df1.iloc[i][c]}"
                
    command = df1.iloc[i]["file"] + command
    
    # current_model = f"logs"
    # pathlib.Path(f'{current_model}/').mkdir(parents=True, exist_ok=True)#metrics

    # windows = False#True
    if platform. uname()[0] == "Windows":
        command = f"python -u {command}"
    else:
        command = f"nohup python3 -u {command} > {current_model}/{fn}.out"
    print(command)
    print()
    # os.system(command)
