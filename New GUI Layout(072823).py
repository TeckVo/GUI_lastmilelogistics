from tkinter import *
from tkinter import filedialog as fd
import webbrowser
from tkinter import messagebox
from tkinter import ttk, filedialog
from tkinter.filedialog import askopenfile
import matplotlib.pyplot as plt
from scipy import interpolate
from PIL import ImageTk, Image
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import xgboost
from datetime import timedelta, datetime
from sklearn.metrics import r2_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, csr_matrix, coo_matrix
from sklearn.cluster import KMeans
import seaborn as sns
import itertools
import time
import math
from statistics import mean 
import copy
import random
from itertools import combinations
import matplotlib.patches as mpatches 
from matplotlib.figure import Figure
import io
import folium
from folium.features import DivIcon
import warnings
warnings.filterwarnings("ignore")
from IPython.display import display
import ipywidgets as widgets
from io import BytesIO
import sys
import os
import random as rn
import requests
import json 
from selenium import webdriver
try:
    import time 
    time.clock = time.time
except:
    print("Unknown error when import time package at main file")
programStart = time.clock()


def get_file_name(file_entry):
    global file_name # inform function to assign it to external/global variable
    file_name = fd.askopenfilename(title="Select file", filetypes=(("CSV Files","*.csv"),))
    file_entry.delete(0, 'end')
    file_entry.insert(0, file_name)
 
file_name = None

root = Tk()  # create root window
root.title("Last-mile logistic system")  # title of the GUI window
root.maxsize(3000, 2500)  # specify the max size the window can expand to
root.config(bg="skyblue")  # specify background color
root.resizable(0, 0)



# Create left and right frames
left_frame1 = Frame(root, width=150, height=150, bg='grey')
left_frame1.grid(row=0, column=0, padx=10, pady=5)


left_frame2 = Frame(root, width=150, height=150, bg='grey')
left_frame2.grid(row=1, column=0, padx=10, pady=5)

left_frame3 = Frame(root, width=150, height=150, bg='grey')
left_frame3.grid(row=2, column=0, padx=10, pady=5)



right_frame1 = Frame(root, width=1030, height=330, bg='grey')
right_frame1.grid(row=0, column=1, padx=10, pady=5)

right_frame2 = Frame(root, width=1030, height=330, bg='grey')
right_frame2.grid(row=1, column=1, rowspan=2, padx=10, pady=5)




Label(left_frame1, text="Upload Data Set", relief=RAISED, font=('times', 12, 'bold')).grid(row=0, column=0, padx=5, pady=5)
Label(left_frame1, text="Input Parameter", relief=RAISED, font=('times', 12, 'bold')).grid(row=2, column=0, padx=5, pady=5)





#Labels in left_frame1
Label_0 = Label(left_frame1, text="Small Scale",relief=RAISED , fg="white", bg="black", font=('times', 12, 'bold'))
Label_0.grid(row=3, column=1, padx=5, pady=8, sticky=W)


Label_01 = Label(left_frame1, text="Large Scale",relief=RAISED , fg="white", bg="black", font=('times', 12, 'bold'))
Label_01.grid(row=3, column=2, padx=5, pady=8, sticky=W)


Label_1 = Label(left_frame1, text="Number of Vehicle",relief=RAISED , fg="white", bg="black", font=('times', 12, 'bold'),width =14)
Label_1.grid(row=4, column=0, padx=5, pady=8, sticky=W)

Label_2 = Label(left_frame1, text="Capacity of Vehicle",relief=RAISED, fg="white", bg="black",font=('times', 12, 'bold'),width =14)
Label_2.grid(row=6, column=0, padx=5, pady=8, sticky=W)

Label_16 = Label(left_frame1, text="Optimal Workload",relief=RAISED, fg="white", bg="black",font=('times', 12, 'bold'),width =14)
Label_16.grid(row=5, column=0, padx=5, pady=8, sticky=W)


# Entry variables in left_frame1
NumberofVehicle = IntVar()
NumberofVehicle.set(2)
NumberofVehicle_Entry = Entry(left_frame1, font=('times', 12, 'bold'), width =7, textvariable=NumberofVehicle).grid(row=4, column=1, padx=5, pady=5)

CapacityofVehicle = IntVar()
CapacityofVehicle.set(0)
CapacityofVehicle_Entry = Entry(left_frame1, font=('times', 12, 'bold'), width =7,textvariable=CapacityofVehicle ).grid(row=6, column=1, padx=5, pady=5)

NumberofVehicle2 = IntVar()
NumberofVehicle2.set(20)
NumberofVehicle2_Entry = Entry(left_frame1, font=('times', 12, 'bold'), width =7, textvariable=NumberofVehicle2).grid(row=4, column=2, padx=5, pady=5)

CapacityofVehicle2 = IntVar()
CapacityofVehicle2.set(0)
CapacityofVehicle2_Entry = Entry(left_frame1, font=('times', 12, 'bold'), width =7,textvariable=CapacityofVehicle2).grid(row=6, column=2, padx=5, pady=5)

# Button upload data set

def Upload_Action():
    newWindow = Toplevel(root)
    newWindow.title("Upload your data")
    entry_csv = Entry(newWindow, text="", width = 50)
    entry_csv.grid(row=0, column =1, sticky = 'w', padx = 5, pady =5)
    Label(newWindow, text="Input CSV file").grid(row=0, column=0, sticky='w')
    
    
    def run_and_close(event=None):
        close()

    def close(event=None):
        newWindow.withdraw() # if you want to bring it back
        newWindow.destroy()
        
    def format_message():
        newWindow_text = Toplevel(newWindow)
        above_frame = Frame(newWindow_text, bg='grey')
        above_frame.grid(row=0, column=0, padx=10, pady=5)
        #show pic in above frame
        image8= Image.open(r"D:/Research Dr. Stanley (MSU)/GUI for Last-mile logistics system/GUI for last-mile logistics system/data format.png")
        image_resize8 = image8.resize((1400,650), Image.ANTIALIAS)
        image_resize8.save('data format_resize.png')
        display = PhotoImage(file='data format_resize.png')
        label_pic = Label(above_frame, width = 1400, height = 650, image = display)
        label_pic.grid(row=1, column=0, padx=5, pady=8, sticky=W)
        label_pic.image=display
        label_pic.place()



        below_frame = Frame(newWindow_text, bg='grey')
        below_frame.grid(row=1, column=0, padx=10, pady=5)
        # download excel file in below frame
        
        def download_file():
            format_data = pd.read_excel(r"D:/Research Dr. Stanley (MSU)/GUI for Last-mile logistics system/GUI for last-mile logistics system/Data format.xlsx")
            try:
                with filedialog.asksaveasfile(mode='w', defaultextension=".csv") as file:
                    format_data.to_csv(file.name)
            except:
               print()
        
        down_file = Button( below_frame, text="Download example file",command = download_file, font=('times', 12, 'bold'))
        down_file.grid(row=2, column=0, padx=5, pady=5)
        down_file.place()
                

    Button(newWindow, text="Browse...", width=10, command=lambda:get_file_name(entry_csv)).grid(row=0, column=2, sticky='w',padx=5)
    Button(newWindow, text="Ok",command=run_and_close, width=10).grid(row=3, column=2, sticky='e', padx=5)
    Button(newWindow, text="Cancel", command=close, width=10).grid(row=3, column=3, sticky='w',padx=5)
    newWindow.bind('<Return>', run_and_close)
    newWindow.bind('<Escape>', close)  
    
    Button(newWindow, text="Read me first",command=format_message, width=10).grid(row=0, column=3, sticky='e', padx=5)
    
  

def Default_Action():
    newWindow_data = Toplevel(root)
    Label(newWindow_data, text = "Please select the data set that you want to input into the system",font=('times', 12, 'bold')).grid(row =1, sticky=W)
    var_small = IntVar()
    Checkbutton(newWindow_data, text = "Small Scale Data", variable = var_small, font=('times', 12, 'bold','italic')).grid(row =2, sticky=W)
    var_large = IntVar()
    Checkbutton(newWindow_data, text = "Large Scale Data", variable = var_large, font=('times', 12, 'bold','italic')).grid(row =3, sticky=W)
    Button(newWindow_data, text = "OK",command =newWindow_data.destroy, font=('times', 12), width =10).grid(row =4, sticky=W, padx=5, pady=5)


Upload_button = Button(left_frame1, text="Upload Data",command=Upload_Action, fg="white", bg="black", font=('times', 12, 'bold'), width =10)
Upload_button.grid(row=1, column=0, padx=5, pady=8, sticky=W)

Default_button = Button(left_frame1,text="Default Data",command = Default_Action,fg="white", bg="black", font=('times', 12, 'bold'), width =10)
Default_button .grid(row=1, column=2, padx=5, pady=8, sticky=W)


#Labels in left_frame3
Label_3 = Label(left_frame3, text="Travel Cost",relief=RAISED , fg="white", bg="black", font=('times', 12, 'bold'),width =10)
Label_3.grid(row=3, column=0, padx=5, pady=8, sticky=W)

Label_4 = Label(left_frame3, text="Failure Cost",relief=RAISED, fg="white", bg="black",font=('times', 12, 'bold'),width =10)
Label_4.grid(row=4, column=0, padx=5, pady=8, sticky=W)

Label_5 = Label(left_frame3, text="Total Cost",relief=RAISED, fg="white", bg="black",font=('times', 12, 'bold'),width =10)
Label_5.grid(row=5, column=0, padx=5, pady=8, sticky=W)


Label_14 = Label(left_frame2, text = "Small Scale", relief=RAISED, fg="white", bg="black",font=('times', 12, 'bold'))
Label_14.grid(row=1, column=0, padx=5, pady=8, sticky=W)

Label_15 = Label(left_frame2, text = "Large Scale", relief=RAISED, fg="white", bg="black",font=('times', 12, 'bold'))
Label_15.grid(row=1, column=1, padx=5, pady=8, sticky=E)



#---------------------------Optimal capacity 
def Optimal_capacity():
    data = pd.read_csv(r"D:\Research Dr. Stanley (MSU)\GUI for Last-mile logistics system\GUI for last-mile logistics system\data_out_modified_40drop_own_final30.csv", nrows=7367)
    #data = data[data['attempt_date']<='2015-10-01']
    data.fillna(-1, inplace=True)
    data.sort_values(by=["route", "route_rank"], inplace=True)
    data_route = data["route"].values
    data_route_rank = data["route_rank"].values
    data_gc_dist = data["gc_dist"].values
    data_lat = data["lat"].values
    data_long = data["long"].values
    great_dist_diff = np.zeros(len(data))
    data["great_dist_diff"] = great_dist_diff
    data["log_greatdistdiff"] = np.log1p(data["great_dist_diff"])
    data["attempthour"] = (data["attempthour"].str[:2].astype(int)/6).astype(int)
    # New customerid_day_frequency 
    customer_days_dict = {}
    customer_id = data["customerid"].values
    attempt_date = data["attempt_date"].values
    for n in range(0, len(data)):
        if customer_id[n] not in customer_days_dict:
            customer_days_dict[customer_id[n]] = set()
        customer_days_dict[customer_id[n]].add(attempt_date[n])
    for customer in customer_days_dict:
        customer_days_dict[customer] = np.array(list(customer_days_dict[customer]))

    customerid_day_frequency_new = np.zeros(len(data))
    for n in range(0, len(data)):
        customerid_day_frequency_new[n] = np.sum(customer_days_dict[customer_id[n]] <= attempt_date[n])
    data["customerid_day_frequency_new"] = customerid_day_frequency_new
    
    # New zipcode_count
    zipcode_dict = {}
    zipcode = data["zipcode"].values
    attempt_date = data["attempt_date"].values
    for n in range(0, len(data)):
        if zipcode[n] not in zipcode_dict:
            zipcode_dict[zipcode[n]] = []
        zipcode_dict[zipcode[n]].append(attempt_date[n])
    for zipc in zipcode_dict:
        zipcode_dict[zipc] = np.array(zipcode_dict[zipc])

    zipcode_count_new = np.zeros(len(data))
    for n in range(0, len(data)):
        zipcode_count_new[n] = np.sum(zipcode_dict[zipcode[n]] <= attempt_date[n])
    data["zipcode_count_new"] = zipcode_count_new
    days = sorted(list(set(data["attempt_date"])))
    
    # Simulate days to compute optimization performance 
    for day in days:
        if day < "2015-02-02":
            continue
        data_day = data[data["attempt_date"]==day].copy(deep=True)
        routes = set(data_day["route"].tolist()) 
        
        for route in routes:
            data_route = data_day[data_day["route"]==route].copy(deep=True)
            data_route.sort_values(by="route_rank", inplace=True, ignore_index=True)  
            drops_in_route = len(data_route)
        
        Optimal_work = round((drops_in_route*2-math.log(sum(data_route_rank))),2)
        
    show_OptimapCapacity.delete(0, 'end')
    show_OptimapCapacity.insert(0, str(Optimal_work))

#Result message in left_frame1

Optimal_capacity_button = Button(left_frame2, command = Optimal_capacity,text="Optimal Workload",font=('times', 12, 'bold'),width =14)
Optimal_capacity_button.grid(row=2, column=0, padx=5, pady=5, sticky=W)

show_OptimapCapacity = Entry(left_frame1, font=('times',12,'bold'), width =7)
show_OptimapCapacity.grid(row=5,column=1, padx=5, pady=5)


#-----------------------------Failure_probability
def Failure_probability():   
    # Set parameters and simulate
    from os.path import exists 
    for s_thresh in range(900,1000,50000):
        route_id_count = 0
        s_thresh /= 1000 
        
    if file_name:
        data = pd.read_csv(file_name)
    else:
        data = pd.read_csv(r"D:\Research Dr. Stanley (MSU)\GUI for Last-mile logistics system\GUI for last-mile logistics system\data_out_modified_40drop_own_final30.csv",nrows=7367)    
    #data = data[data['attempt_date']<='2015-10-01']
    data.fillna(-1, inplace=True)
    data.sort_values(by=["route", "route_rank"], inplace=True)

    # Construct some features used in the models 
    data["attempthour"] = (data["attempthour"].str[:2].astype(int)/6).astype(int)

    # New customerid_day_frequency 
    customer_days_dict = {}
    customer_id = data["customerid"].values
    attempt_date = data["attempt_date"].values
    for n in range(0, len(data)):
        if customer_id[n] not in customer_days_dict:
            customer_days_dict[customer_id[n]] = set()
        customer_days_dict[customer_id[n]].add(attempt_date[n])
    for customer in customer_days_dict:
        customer_days_dict[customer] = np.array(list(customer_days_dict[customer]))

    customerid_day_frequency_new = np.zeros(len(data))
    for n in range(0, len(data)):
        customerid_day_frequency_new[n] = np.sum(customer_days_dict[customer_id[n]] <= attempt_date[n])
    data["customerid_day_frequency_new"] = customerid_day_frequency_new

    # New zipcode_count
    zipcode_dict = {}
    zipcode = data["zipcode"].values
    attempt_date = data["attempt_date"].values
    for n in range(0, len(data)):
        if zipcode[n] not in zipcode_dict:
            zipcode_dict[zipcode[n]] = []
        zipcode_dict[zipcode[n]].append(attempt_date[n])
    for zipc in zipcode_dict:
        zipcode_dict[zipc] = np.array(zipcode_dict[zipc])

    zipcode_count_new = np.zeros(len(data))
    for n in range(0, len(data)):
        zipcode_count_new[n] = np.sum(zipcode_dict[zipcode[n]] <= attempt_date[n])
    data["zipcode_count_new"] = zipcode_count_new

    use_features = ["own_other", "route_rank", "shipments_route", "gc_dist"]
    use_features += ["shipmentamount", "shipmentweight", "Average_Temperature", "crime_month_count", "hh_size", "avg_income", "pop_densit"]
    use_features += ["att_weekday"]
    use_features += ["customerid"]
    use_features += ["customerid_day_frequency_new", "customer_day_packages", "failure_rate_date_lag1", "failure_rate_date_ma_week1", "deliveries_date_ma_week1", "days_since_last_delivery", "days_since_last_delivery_failure"]
    use_features += ["zipcode", "zipcode_count_new", "zipcode_day_count", "failure_rate_date_zipcode_lag1", "day_zipcode_frequency", "retail_dens", "literate_hoh"]
    use_features += ["lat_diff_abs", "long_diff_abs"]
    use_features += ["att_mnth", "dropsize"]

    data["lat_diff_abs"].fillna(-1, inplace=True)
    data["long_diff_abs"].fillna(-1, inplace=True)

    days = sorted(list(set(data["attempt_date"])))


    # Simulate days to compute optimization performance 
    for day in days:
        if day < "2015-02-02":
            continue
        data_day = data[data["attempt_date"]==day].copy(deep=True)
       
        # Select model
        model_type = "lgb"
        
        if model_type == "lgb":
            depth = 2
            feat_frac=0.3
            subsample_frac =0.6
            num_iter = 200
            if not exists ('gbm_model_save.txt'):
                continue
            gbm = lgb.Booster(model_file='gbm_model_save.txt')
        
        if model_type =="lgb":
            data_day["pred_failure"] = gbm.predict(data_day[use_features], num_threads=1)
        else:
            data_day["pred_failure"] = gbm.predict_proba(data_day[use_features])[:, 1] 
        probability_frame = pd.DataFrame()
        probability_frame["customerid"] = data_day["customerid"]
        probability_frame["pred_failure"] = round(data_day["pred_failure"]*100,2)
        bargraph = probability_frame.plot.bar(x='customerid',y='pred_failure')
        func = lambda y, pos: f"{int(y)}%"
        bargraph.yaxis.set_major_formatter(func)
        plt.title('Failure delivery probability for each customerid')
        plt.xlabel('Customer ID')
        plt.ylabel('Failure probability')
        
        # open saved pic 
        plt.savefig('Failure_probability.png')
        image_open1 = Image.open('Failure_probability.png')
        image_resize1 = image_open1.resize((310,290), Image.ANTIALIAS)
        image_resize1.save('Failure_probability_resize.png')
        image1= PhotoImage(file='Failure_probability_resize.png')
        image1.subsample(1,1)
        show_image1 = Canvas(right_frame1,width=310, height=290)
        show_image1.grid(row=0,column=0, padx=5, pady=5)
        show_image1.place(relx=0, rely=0)
        show_image1.create_image(0,0, image=image1, anchor="nw")
        show_image1.image = image1
       
    def plot_show1():
        plt.show()
        plt.close()
        
    plot_button1 = Button(right_frame1, text="Zoom",command =plot_show1, font=('times', 12, 'bold'))
    plot_button1.grid(row=2, column=0, padx=5, pady=5)
    plot_button1.place(relx=0, rely=0.9)
    
    def save_data():
        small_scale = pd.DataFrame()
        small_scale["customerid"] = data_day["customerid"]
        small_scale["pred_failure"] = data_day["pred_failure"]
        small_scale["lat"] = data_day["lat"]
        small_scale["long"] = data_day["long"]
        
        try:
            with filedialog.asksaveasfile(mode='w', defaultextension=".csv") as file:
                small_scale.to_csv(file.name)
        except:
           print()
       
    down_button1 = Button(right_frame1, text="Save data",command = save_data, font=('times', 12, 'bold'))
    down_button1.grid(row=2, column=0, padx=5, pady=5)
    down_button1.place(relx=0.07, rely=0.9)
    
    def destroy_1():
        show_image1.destroy()
        plot_button1.destroy()
        destroy_button1.destroy()
        down_button1.destroy()
    
    destroy_button1 = Button(right_frame1, text="Exit",command =destroy_1, font=('times', 12, 'bold'))
    destroy_button1.grid(row=2, column=0, padx=5, pady=5)
    destroy_button1.place(relx=0.17, rely=0.9)
    
probability_button = Button(left_frame2, text="Failure Probability",font=('times', 12, 'bold'), command =Failure_probability, width =14)
probability_button.grid(row=3, column=0, padx=5, pady=5, stick = W)


#---------------------------------CVRP Model
def CVRP_NEW():
    
    def min_objective_greedy_with_return_with_random_euc(OC_lat_pos, OC_long_pos, route_data):
        
        route_drops = []
        route_dist = []
        route_set = set()

        base_pred_success_mat = np.zeros((len(route_data), len(route_data)))
        base_pred_dist_mat = np.zeros((len(route_data), len(route_data)))
        pos_success_dict = {}
        pos_dist_dict = {}
        for pos in route_data["route_rank"].astype(int).values:
            base_df = route_data.copy(deep=True)
            base_df["route_rank"] = pos
            if len(route_data) > 1:
                pairwise_lat = np.array(list(combinations(route_data["lat"].tolist(), 2)))
                pairwise_long = np.array(list(combinations(route_data["long"].tolist(), 2)))
                avg_pairwise_lat_diff = np.average(np.abs(pairwise_lat[:, 0] - pairwise_lat[:, 1]))
                avg_pairwise_long_diff = np.average(np.abs(pairwise_long[:, 0] - pairwise_long[:, 1]))
            else:
                avg_pairwise_lat_diff = 0
                avg_pairwise_long_diff = 0
            base_df["lat_diff_abs"] = avg_pairwise_lat_diff
            base_df["long_diff_abs"] = avg_pairwise_long_diff
            if model_type=="lgb":
                base_pred_success = 1 - gbm.predict(base_df[use_features], num_threads=1)
            else:
                base_pred_success = 1 - gbm.predict_proba(base_df[use_features])[:, 1]
            base_pred_dist = np.array(base_df["lat_diff_abs"]*DIST_COST_LAT + base_df["long_diff_abs"]*DIST_COST_LONG)
            base_pred_success_mat[:, pos-1] = base_pred_success
            base_pred_dist_mat[:, pos-1] = np.array(base_pred_dist)
            pos_success_dict[pos] = base_pred_success
            pos_dist_dict[pos] = base_pred_dist
        base_pred_success = 0*base_pred_success
        base_pred_dist = 0*base_pred_dist
        for pos in pos_success_dict:
            base_pred_success += pos_success_dict[pos]
            base_pred_dist += pos_dist_dict[pos]
        base_pred_success /= len(pos_success_dict)
        base_pred_dist /= len(pos_dist_dict)
        base_pred_success = np.mean(base_pred_success_mat, axis=1)
        base_pred_dist = np.mean(base_pred_dist_mat, axis=1)
            
        lat_vals = route_data["lat"].values
        long_vals = route_data["long"].values
         
        while len(route_drops) < len(lat_vals):
            min_obj = 1000000000000
            max_node = None
            SUCCESS_WEIGHT_START=1
            SUCCESS_WEIGHT_MID=1
            if len(route_drops) == 0:                
                route_data["route_rank"] = 1
                route_data["lat_diff_abs"] = np.abs(OC_lat_pos - route_data["lat"])
                route_data["long_diff_abs"] = np.abs(OC_long_pos - route_data["long"])
                route_data["log_greatdistdiff"] = route_data.apply(lambda x: np.log1p(great_distance(OC_lat_pos, OC_long_pos, x["lat"], x["long"])), axis=1) # Great distance
                if model_type=="lgb":
                    pred_success = 1 - gbm.predict(route_data[use_features], num_threads=1)
                else:
                    pred_success = 1 - gbm.predict_proba(route_data[use_features])[:, 1]
                pred_success_diff = base_pred_success - pred_success
                dist_diff = np.abs(OC_lat_pos - route_data["lat"])*DIST_COST_LAT + np.abs(OC_long_pos - route_data["long"])*DIST_COST_LONG
                obj_vals = dist_diff + (SUCCESS_WEIGHT_START*pred_success_diff) * FAIL_COST
                for n in range(0, len(pred_success_diff)):
                    if n not in route_set:
                        if obj_vals[n] < min_obj:
                            min_obj = obj_vals[n]
                            max_node = n
                route_set.add(max_node)
                route_drops.append(max_node)
                route_dist.append(great_distance(OC_lat_pos, OC_long_pos, lat_vals[max_node], long_vals[max_node])) # Great distance
                prev_lat = lat_vals[max_node]
                prev_long = long_vals[max_node]
            else:
                route_data["route_rank"] = len(route_drops) + 1
                route_data["lat_diff_abs"] = np.abs(prev_lat - route_data["lat"])
                route_data["long_diff_abs"] = np.abs(prev_long - route_data["long"])
                route_data["log_greatdistdiff"] = route_data.apply(lambda x: np.log1p(great_distance(prev_lat, prev_long, x["lat"], x["long"])), axis=1) # Great distance
                if model_type=="lgb":
                    pred_success = 1 - gbm.predict(route_data[use_features], num_threads=1)
                else:
                    pred_success = 1 - gbm.predict_proba(route_data[use_features])[:, 1]
                pred_success_diff = base_pred_success - pred_success
                dist_diff = np.abs(prev_lat - route_data["lat"])*DIST_COST_LAT + np.abs(prev_long - route_data["long"])*DIST_COST_LONG          
                obj_vals = dist_diff + (SUCCESS_WEIGHT_MID*pred_success_diff) * FAIL_COST
                for n in range(0, len(pred_success_diff)):
                    if n not in route_set:
                        if obj_vals[n] < min_obj:
                            min_obj = obj_vals[n]
                            max_node = n
                route_set.add(max_node)
                route_drops.append(max_node)
                route_dist.append(great_distance(prev_lat, prev_long, lat_vals[max_node], long_vals[max_node])) # Great distance
                prev_lat = lat_vals[max_node]
                prev_long = long_vals[max_node]    
        
        route_dist.append(great_distance(OC_lat_pos, OC_long_pos, prev_lat, prev_long)*DIST_COST_GREAT) # Great distance
                
        rand_threshold = 0 # Driver deviation
        rand_nums = np.random.randint(1000, size=len(route_drops))
        route_dist = []
        
        for n in range(0, len(rand_nums)):
            if rand_nums[n] < rand_threshold and len(rand_nums) > 1:
                curr_drop = route_drops[n] 
                route_drops = route_drops[:n]+route_drops[n+1:]
                new_drop_pos = np.random.choice(route_drops, 1)[0]
                route_drops.insert(new_drop_pos, curr_drop)         
                
        route_dist.append(great_distance(OC_lat_pos, OC_long_pos, lat_vals[route_drops[0]], long_vals[route_drops[0]])*DIST_COST_GREAT) # Great distance
        for n in range(0, len(route_drops)-1):  
            route_dist.append(great_distance(lat_vals[route_drops[n]], long_vals[route_drops[n]], lat_vals[route_drops[n+1]], long_vals[route_drops[n+1]])*DIST_COST_GREAT) # Great distance
        
        route_dist.append(great_distance(OC_lat_pos, OC_long_pos, lat_vals[route_drops[-1]], long_vals[route_drops[-1]])*DIST_COST_GREAT) # Great distance
        route_seq_dict = {"route_drops":np.array(route_drops), "route_dist":np.array(route_dist)}
                         
        return route_seq_dict

    # Function to calculate total route (great) distance
    data = pd.read_csv(r"D:\Research Dr. Stanley (MSU)\GUI for Last-mile logistics system\GUI for last-mile logistics system\data_out_modified_40drop_own_final30.csv",nrows=21832)
    def great_distance(x1, y1, x2, y2):
        
        if x1 == x2 and y1 == y2:
            return 0

        x1 = math.radians(x1)
        y1 = math.radians(y1)
        x2 = math.radians(x2)
        y2 = math.radians(y2)

        angle1 = math.acos(math.sin(x1) * math.sin(x2) + math.cos(x1) * math.cos(x2) * math.cos(y1 - y2))
        angle1 = math.degrees(angle1)

        return 60.0 * angle1

    # Calculate the great route distance for the existing dataset based on the original sequencing
    routedistance_dict = {}
    for route in set(data["route"].tolist()):
        route_vals = data[["route_rank", "lat", "long"]].loc[data["route"] == route].sort_values(by="route_rank").values
        total_dist = 0
        for n in range(len(route_vals)-1):
            total_dist += great_distance(route_vals[n, 1], route_vals[n, 2], route_vals[n+1, 1], route_vals[n+1, 2])
        routedistance_dict[route] = total_dist
        
    # Set parameters and simulate
    from os.path import exists 
    for s_thresh in range(900,1000,50000):
        route_id_list = []
        route_len_list = []
        route_dist_cost_actual_list = []
        route_dist_cost_best_objective_list = []
        route_failure_cost_actual_list = []
        route_failure_cost_best_objective_list = []
        route_total_cost_actual_list = []
        route_total_cost_best_objective_list = []
        

        route_id_count = 0
        s_thresh /= 1000 
        if file_name:
            data = pd.read_csv(file_name)
        else:
            data = pd.read_csv(r"D:\Research Dr. Stanley (MSU)\GUI for Last-mile logistics system\GUI for last-mile logistics system\data_out_modified_40drop_own_final30.csv",nrows=21832)    
        
        data.fillna(-1, inplace=True)
        data.sort_values(by=["route", "route_rank"], inplace=True)
        
        # Construct some features used in the models 
        routedistance_dict = {}
        for route in set(data["route"].tolist()):
            route_vals = data[["route_rank", "lat", "long"]].loc[data["route"] == route].sort_values(by="route_rank").values
            total_dist = 0
            for n in range(len(route_vals)-1):
                total_dist += great_distance(route_vals[n, 1], route_vals[n, 2], route_vals[n+1, 1], route_vals[n+1, 2])
            routedistance_dict[route] = total_dist

        data["totalroutedistance"] = data["route"].apply(lambda x: routedistance_dict[x])
        data["log_gc_dist"] = np.log1p(data["gc_dist"])
        data["log_shipmentamount"] = np.log1p(data["shipmentamount"])
        data["log_shipmentweight"] = np.log1p(data["shipmentweight"])
        data["log_totalroutedistance"] = np.log1p(data["totalroutedistance"])

        data_route = data["route"].values
        data_route_rank = data["route_rank"].values
        data_gc_dist = data["gc_dist"].values
        data_lat = data["lat"].values
        data_long = data["long"].values
        great_dist_diff = np.zeros(len(data))
        for n in range(0, len(data)):
            if n == 0 or data_route[n] != data_route[n-1]:
                great_dist_diff[n] = data_gc_dist[n]
            else:
                great_dist_diff[n] = great_distance(data_lat[n], data_long[n], data_lat[n-1], data_long[n-1])
        data["great_dist_diff"] = great_dist_diff
        data["log_greatdistdiff"] = np.log1p(data["great_dist_diff"])
        data["attempthour"] = (data["attempthour"].str[:2].astype(int)/6).astype(int)
        
        # New customerid_day_frequency 
        customer_days_dict = {}
        customer_id = data["customerid"].values
        attempt_date = data["attempt_date"].values
        for n in range(0, len(data)):
            if customer_id[n] not in customer_days_dict:
                customer_days_dict[customer_id[n]] = set()
            customer_days_dict[customer_id[n]].add(attempt_date[n])
        for customer in customer_days_dict:
            customer_days_dict[customer] = np.array(list(customer_days_dict[customer]))

        customerid_day_frequency_new = np.zeros(len(data))
        for n in range(0, len(data)):
            customerid_day_frequency_new[n] = np.sum(customer_days_dict[customer_id[n]] <= attempt_date[n])
        data["customerid_day_frequency_new"] = customerid_day_frequency_new
        
        # New zipcode_count
        zipcode_dict = {}
        zipcode = data["zipcode"].values
        attempt_date = data["attempt_date"].values
        for n in range(0, len(data)):
            if zipcode[n] not in zipcode_dict:
                zipcode_dict[zipcode[n]] = []
            zipcode_dict[zipcode[n]].append(attempt_date[n])
        for zipc in zipcode_dict:
            zipcode_dict[zipc] = np.array(zipcode_dict[zipc])

        zipcode_count_new = np.zeros(len(data))
        for n in range(0, len(data)):
            zipcode_count_new[n] = np.sum(zipcode_dict[zipcode[n]] <= attempt_date[n])
        data["zipcode_count_new"] = zipcode_count_new
        
        use_features = ["own_other", "route_rank", "shipments_route", "gc_dist"]
        use_features += ["shipmentamount", "shipmentweight", "Average_Temperature", "crime_month_count", "hh_size", "avg_income", "pop_densit"]
        use_features += ["att_weekday"]
        use_features += ["customerid"]
        use_features += ["customerid_day_frequency_new", "customer_day_packages", "failure_rate_date_lag1", "failure_rate_date_ma_week1", "deliveries_date_ma_week1", "days_since_last_delivery", "days_since_last_delivery_failure"]
        use_features += ["zipcode", "zipcode_count_new", "zipcode_day_count", "failure_rate_date_zipcode_lag1", "day_zipcode_frequency", "retail_dens", "literate_hoh"]
        use_features += ["lat_diff_abs", "long_diff_abs"]
        use_features += ["att_mnth", "dropsize"]

        data["lat_diff_abs"].fillna(-1, inplace=True)
        data["long_diff_abs"].fillna(-1, inplace=True)
        
        # Compute distance from each drop to OC
        OC_lat = np.array([-23.558123100, -23.433411500, -23.558123100, -23.514664500, -23.526479600, -23.661295500, -23.671398600, -23.492313700])
        OC_long = np.array([-46.609302200, -46.554093100, -46.609302200, -46.653909400, -46.766178900, -46.487164700, -46.716471400, -46.842962500])
        OC_dict = {}
        for OC in set(data["OC"].tolist()):
            data_OC = data[["lat", "long"]].loc[data["OC"]==OC]
            for n in range(0, len(OC_lat)):
                lat_dist = np.abs(data_OC["lat"] - OC_lat[n])
                long_dist = np.abs(data_OC["long"] - OC_long[n])

        OC_dict = {"OC01":3, "OC02":5, "OC03":1, "OC04":0, "OC05":6, "OC06":4, "OC10":2, "OC09":7}
        data["lat_diff_OC"] = data["OC"].apply(lambda x: OC_lat[OC_dict[x]])
        data["long_diff_OC"] = data["OC"].apply(lambda x: OC_long[OC_dict[x]])
        
        # Parameters for travel cost and failure cost 
        TRAVEL_COST_MULTIPLIER = 1
        TRAVEL_SPEED = 30
        DRIVER_WAGE = 1
        DIST_COST_LAT = (0.58/15/1.61 + DRIVER_WAGE/TRAVEL_SPEED)*110.57*TRAVEL_COST_MULTIPLIER
        DIST_COST_LONG = (0.58/15/1.61 + DRIVER_WAGE/TRAVEL_SPEED)*102.05*TRAVEL_COST_MULTIPLIER
        DIST_COST_GREAT = (0.58/15/1.61 + DRIVER_WAGE/TRAVEL_SPEED)/1.852*TRAVEL_COST_MULTIPLIER
        FAIL_COST = CapacityofVehicle.get()
        
        missed_thresholds = 0
        num_drops_total = 0
        
        total_distance_actual = 0
        total_distance_best_success_objective = 0
        
        success_actual_list = []
        success_best_success_objective_list = []
        
        total_cost_actual = 0
        total_cost_best_success_objective = 0
        
        days = sorted(list(set(data["attempt_date"])))
        use_vars = ["attempt_date", "route", "OC", "lat", "long"] + use_features
        weight_list = []
        
        # Simulate days to compute optimization performance 
        for day in days:
            if day < "2015-04-02":
                continue
            data_day = data[data["attempt_date"]==day].copy(deep=True)
            
            num_drops_day = 0
            total_distance_actual_day = 0
            total_distance_best_success_objective_day = 0
            success_actual_day_list = []
            success_best_success_objective_day_list = []
            total_cost_actual_day = 0
            total_cost_best_success_objective_day = 0
            
            
            # Select model
            model_type = "lgb"
            
            if model_type == "lgb":
                depth = 2
                feat_frac=0.3
                subsample_frac =0.6
                num_iter = 200
                if not exists ('gbm_model_save.txt'):
                    continue
                gbm = lgb.Booster(model_file='gbm_model_save.txt')
            
            if model_type =="lgb":
                data_day["pred_success"] = 1 - gbm.predict(data_day[use_features], num_threads=1)
            else:
                data_day["pred_success"] = 1 - gbm.predict_proba(data_day[use_features])[:, 1] 
            
            routes = set(data_day["route"].tolist()) 
            
            for route in routes:
                data_route = data_day[data_day["route"]==route].copy(deep=True)
                data_route.sort_values(by="route_rank", inplace=True, ignore_index=True)  
                OC_ind = data_route["OC"].values[0]
                average_route_success_actual = np.average(data_route["pred_success"])
                total_route_distance_actual = data_route["great_dist_diff"].sum() * DIST_COST_GREAT
                
                # Lowest objective greedy
                #drops_in_route = len(route_drops)
                best_success_route_objective_df = data_route.copy(deep=True)
                best_success_route_objective_dict = min_objective_greedy_with_return_with_random_euc(OC_lat[OC_dict[OC_ind]], OC_long[OC_dict[OC_ind]], best_success_route_objective_df)
                best_success_route_objective_sequence = best_success_route_objective_dict["route_drops"]
                new_route_sequence = np.zeros(len(best_success_route_objective_df))
                for n in range(0, len(best_success_route_objective_sequence)):
                    new_route_sequence[best_success_route_objective_sequence[n]] = n+1
                best_success_route_objective_df["route_rank"] = new_route_sequence
                best_success_route_objective_df = best_success_route_objective_df.sort_values(by="route_rank", ignore_index=True)

                best_success_route_objective_df["lat_diff"] = best_success_route_objective_df["lat"].diff()
                best_success_route_objective_df["long_diff"] = best_success_route_objective_df["long"].diff()
                best_success_route_objective_df["lat_diff_abs"] = np.abs(best_success_route_objective_df["lat_diff"])
                best_success_route_objective_df["long_diff_abs"] = np.abs(best_success_route_objective_df["long_diff"])

                best_success_route_objective_df["lat_diff_abs"].fillna(-1, inplace=True)
                best_success_route_objective_df["long_diff_abs"].fillna(-1, inplace=True)

                best_success_route_objective_df["log_greatdistdiff"] = best_success_route_objective_df.apply(lambda x: np.log1p(great_distance(x["lat"], x["long"], x["lat"]+x["lat_diff"], x["long"]+x["long_diff"])) 
                                                                                  if x["long_diff_abs"]!=-1 else np.log1p(x["gc_dist"]), axis=1) # Great distance
                if model_type =="lgb":
                    best_success_route_objective_success = 1 - gbm.predict(best_success_route_objective_df[use_features], num_threads=1)
                else:
                    best_success_route_objective_success = 1 - gbm.predict_proba(best_success_route_objective_df[use_features])[:, 1]
                best_success_route_objective_df["pred_success"] = best_success_route_objective_success
                average_best_success_route_objective_success = np.average(best_success_route_objective_success)
                total_best_success_route_objective_distance = np.sum(best_success_route_objective_dict["route_dist"])   
                
                # Only consider routes with at least 1 drop 
                
                drops_in_route = len(data_route)
                if drops_in_route > 0:
                        num_drops_day += len(data_route)
                        total_distance_actual_day += total_route_distance_actual
                        total_distance_best_success_objective_day += total_best_success_route_objective_distance
                        success_actual_day_list += data_route["pred_success"].tolist()
                        success_best_success_objective_day_list += best_success_route_objective_df["pred_success"].tolist()
                        total_cost_actual_day +=  total_route_distance_actual + np.sum(1 - data_route["pred_success"].values) * FAIL_COST
                        total_cost_best_success_objective_day +=  total_best_success_route_objective_distance + np.sum(1 - best_success_route_objective_df["pred_success"].values) * FAIL_COST
                   
                route_id_count += 1
                route_id_list.append(route_id_count)
                route_len_list.append(len(data_route))
                
                route_dist_cost_actual_list.append(total_route_distance_actual)
                route_dist_cost_best_objective_list.append(total_best_success_route_objective_distance)
                
                route_failure_cost_actual_list.append(np.sum(1 - data_route["pred_success"].values) * FAIL_COST)
                route_failure_cost_best_objective_list.append(np.sum(1 - best_success_route_objective_df["pred_success"].values) * FAIL_COST)
                
                route_total_cost_actual_list.append(total_route_distance_actual + np.sum(1 - data_route["pred_success"].values) * FAIL_COST)
                route_total_cost_best_objective_list.append(total_best_success_route_objective_distance+ np.sum(1 - best_success_route_objective_df["pred_success"].values) * FAIL_COST)



            num_drops_total += num_drops_day
            total_distance_actual += total_distance_actual_day
            total_distance_best_success_objective += total_distance_best_success_objective_day
            success_actual_list += success_actual_day_list
            success_best_success_objective_list += success_best_success_objective_day_list
            total_cost_actual += total_cost_actual_day
            total_cost_best_success_objective += total_cost_best_success_objective_day  

                
               
        round_total_distance_actual = round(total_distance_actual,2)
        #print(round_total_distance_actual)
        show_round_total_distance_actual_small.delete(0, 'end')
        show_round_total_distance_actual_small.insert(0, str("$")+str(round_total_distance_actual))
       
        
        round_total_distance_best_success_objective = round(total_distance_best_success_objective,2)
        #print(round_total_distance_best_success_objective )
        show_total_distance_best_success_objective_small.delete(0, 'end')
        show_total_distance_best_success_objective_small.insert(0, str("$")+str(round_total_distance_best_success_objective))
       
        
        total_fail_cost_actual = FAIL_COST*num_drops_total*(1-np.average(success_actual_list))
        round_total_fail_cost_actual = round( total_fail_cost_actual,2)
        #print( round_total_fail_cost_actual)
        show_round_total_fail_cost_actual_small.delete(0, 'end')
        show_round_total_fail_cost_actual_small.insert(0, str("$")+str(round_total_fail_cost_actual))
       
        
        
        total_fail_cost_best_success_objective = (FAIL_COST)*num_drops_total*(1-np.average(success_best_success_objective_list))
        round_total_fail_cost_best_success_objective  = round(total_fail_cost_best_success_objective,2)
        #print(round_total_fail_cost_best_success_objective)
        show_total_fail_cost_best_success_objective_small.delete(0, 'end')
        show_total_fail_cost_best_success_objective_small.insert(0, str("$")+str(round_total_fail_cost_best_success_objective))
        
        
        round_total_cost_actual = round(total_cost_actual,2)
        #print(round_total_cost_actual)
        show_round_total_cost_actual_small.delete(0, 'end')
        show_round_total_cost_actual_small.insert(0, str("$")+str( round_total_cost_actual))
        
        
        round_total_cost_best_success_objective = round(total_cost_best_success_objective,2)
        #print(round_total_cost_best_success_objective)
        show_total_cost_best_success_objective_small.delete(0, 'end')
        show_total_cost_best_success_objective_small.insert(0, str("$")+str(round_total_cost_best_success_objective))
        
        # Difference
        diff_total_distance_cost = round(((round_total_distance_best_success_objective-round_total_distance_actual)/round_total_distance_actual)*100,2)
        show_diff_total_distance_cost_small.delete(0, 'end')
        show_diff_total_distance_cost_small.insert(0, str( diff_total_distance_cost)+str("%"))
        
        diff_total_fail_cost = round(((round_total_fail_cost_best_success_objective-round_total_fail_cost_actual)/round_total_fail_cost_actual)*100,2)
        show_diff_total_fail_cost_small.delete(0, 'end')
        show_diff_total_fail_cost_small.insert(0, str( diff_total_fail_cost)+str("%"))
        
        diff_total_cost = round(((round_total_cost_best_success_objective-round_total_cost_actual)/round_total_cost_actual)*100,2)
        show_diff_total_cost_small.delete(0, 'end')
        show_diff_total_cost_small.insert(0, str(diff_total_cost)+str("%"))
        
       
    
        #--------------------Bar gragh
        X_set = ['Travel Cost', 'Failure Cost', 'Total Cost']
        Y_actual = [round_total_distance_actual, round_total_fail_cost_actual, round_total_cost_actual]
        Y_best = [round_total_distance_best_success_objective, round_total_fail_cost_best_success_objective, round_total_cost_best_success_objective]
        
        X_axis = np.arange(len(X_set))
        
        plt.bar(X_axis - 0.2, Y_actual, 0.4, label = 'Baseline')
        for i in range(len(X_axis)):
            plt.text(i,Y_actual[i],Y_actual[i], ha='right',color='green', fontweight='bold')
        
        plt.bar(X_axis + 0.2, Y_best, 0.4, label = 'Optimal')
        for i in range(len(X_axis)):
            plt.text(i,Y_best[i],Y_best[i], ha='left',color='red', fontweight='bold')
        
        plt.xticks(X_axis, X_set)
        plt.xlabel("Cost groups")
        plt.ylabel("Cost ($)")
        plt.title("Costs comparison of CVRP model with/without failure probability")
        plt.legend()
        plt.savefig('Small_scale_comparison.png')
        
        #------------Open saved pic
        image_open3 = Image.open('Small_scale_comparison.png')
        image_resize3 = image_open3.resize((310,290),Image.ANTIALIAS)
        image_resize3.save('Small_scale_comparison_resize.png')
        image3 = PhotoImage(file='Small_scale_comparison_resize.png')
        image3.subsample(1,1)
        show_image3 = Canvas(right_frame1,width=310, height=290)
        show_image3.grid(row=1,column=0, columnspan=2, padx=5, pady=5)
        show_image3.place(relx=0.35, rely=0)
        show_image3.create_image(0,0, image=image3, anchor="nw")
        show_image3.image = image3     
    def plot_show3():
        plt.show()
        plt.close()
        
    plot_button3 = Button(right_frame1, text="Zoom",command =plot_show3, font=('times', 12, 'bold'))
    plot_button3.grid(row=2, column=0, padx=5, pady=5)
    plot_button3.place(relx=0.35, rely=0.9)
    
    def save_data2():
        cost_perf_df = pd.DataFrame()
        cost_perf_df["route_id"] = route_id_list
        cost_perf_df["route_len"] = route_len_list
        cost_perf_df["route_dist_cost_actual"] = route_dist_cost_actual_list
        cost_perf_df["route_dist_cost_best_objective"] = route_dist_cost_best_objective_list
        cost_perf_df["route_failure_cost_actual"] = route_failure_cost_actual_list
        cost_perf_df["route_failure_cost_best_objective"] = route_failure_cost_best_objective_list
        cost_perf_df["route_total_cost_actual"] = route_total_cost_actual_list
        cost_perf_df["route_total_cost_best_objective"] = route_total_cost_best_objective_list
        #cost_perf_df.to_csv("cost_perf_small.csv", index=False) 
        
        try:
            with filedialog.asksaveasfile(mode='w', defaultextension=".csv") as file:
                cost_perf_df.to_csv(file.name)
        except:
           print()
           
        route_perf_df = pd.DataFrame()
        route_perf_df["customerid"] = data_day["customerid"]
        route_perf_df["lat"] = data_day["lat"]
        route_perf_df["long"] = data_day["long"]
        route_perf_df["route"] = data_day["route"]
        route_perf_df.to_csv("route_perf_small_scale.csv", index=False)
           
    down_button2 = Button(right_frame1, text="Save data",command = save_data2, font=('times', 12, 'bold'))
    down_button2.grid(row=2, column=0, padx=5, pady=5)
    down_button2.place(relx=0.42, rely=0.9)
    
    def destroy_3():
        show_image3.destroy()
        plot_button3.destroy()
        destroy_button3.destroy()
        down_button2.destroy()
    
    destroy_button3 = Button(right_frame1, text="Exit",command =destroy_3, font=('times', 12, 'bold'))
    destroy_button3.grid(row=2, column=0, padx=5, pady=5)
    destroy_button3.place(relx=0.52, rely=0.9)
    
    
    # Map route 
    def get_directions_response(lat1, long1, lat2, long2, mode='drive'):
        url = "https://route-and-directions.p.rapidapi.com/v1/routing"
        key = "165aea8a7fmsh8e6392177d6e3a5p1d9590jsn08d0be6112ff"
        host = "route-and-directions.p.rapidapi.com"
        headers = {"X-RapidAPI-Key": key, "X-RapidAPI-Host": host}
        querystring = {"waypoints":f"{str(lat1)},{str(long1)}|{str(lat2)},{str(long2)}","mode":mode}
        response = requests.request("GET", url, headers=headers, params=querystring)
        return response

    location = pd.read_csv(r"D:/Research Dr. Stanley (MSU)/GUI for Last-mile logistics system/GUI for last-mile logistics system/route_perf_small_scale.csv")

    location = location[["lat", "long"]]
    location_list = location.values.tolist()
    new_list = []
    for i in location_list:
       new_list.append(tuple(i))

    # Depot 
    df_depot = pd.DataFrame()
    depot_lat = [-23.514664500, -23.661295500, -23.433411500, -23.558123100, -23.671398600,-23.526479600,-23.492313700,-23.558123100]
    depot_long = [-46.653909400, -46.487164700, -46.554093100,-46.609302200, -46.716471400,-46.766178900, -46.842962500,-46.609302200]
    depot_name = ["OC01", "OC02", "OC03", "OC04", "OC05", "OC06", "OC09", "OC10"]
    df_depot['depot_name']= depot_name
    df_depot['depot_lat']= depot_lat
    df_depot['depot_long']= depot_long
    depot_location = df_depot[["depot_lat","depot_long"]]
    depot_location_list = depot_location.values.tolist()
    new_depot_list=[]
    for i in depot_location_list:
       new_depot_list.append(tuple(i))

    final_list =  new_list + new_depot_list

    m = folium.Map()
    colors = ['blue','red','green','black','maroon','orange', 'gold']
    for point in new_list[:]:
        folium.Marker(point,icon=folium.Icon( color='blue')).add_to(m)
        
    for point in new_depot_list[:]:
        folium.Marker(point,icon=folium.Icon( color='red')).add_to(m)
       
    routes = [[14,0,1,2,3,4,5,6,7,8,9,10,14]]
    route_trans = []
    for i in range(len(routes)):
        trans = []
        for shipment_index in routes[i]:
            trans.append(final_list[shipment_index]) 
        route_trans.append(trans) 
        
        
    responses = []
    for r in range(len( route_trans)):
        for n in range(len(route_trans[r])-1):
            lat1 = route_trans[r][n][0]
            lon1 = route_trans[r][n][1]
            lat2 = route_trans[r][n+1][0]
            lon2 = route_trans[r][n+1][1]
            response= get_directions_response(lat1, lon1, lat2, lon2, mode='drive')
            responses.append(response)
            mls = response.json()['features'][0]['geometry']['coordinates']
            points = [(i[1], i[0]) for i in mls[0]]
            folium.PolyLine(points, weight=5, opacity=1, color=colors[r]).add_to(m)
            temp = pd.DataFrame(mls[0]).rename(columns={0:'Lon', 1:'Lat'})[['Lat', 'Lon']]
            df = pd.DataFrame()
            df = pd.concat([df, temp])
            sw = df[['Lat', 'Lon']].min().values.tolist()
            sw = [sw[0]-0.0005, sw[1]-0.0005]
            ne = df[['Lat', 'Lon']].max().values.tolist()
            ne = [ne[0]+0.0005, ne[1]+0.0005]
            m.fit_bounds([sw, ne])   
                 
    # Save map
    mapFname = 'route_map_small.html'
    m.save(mapFname)
    mapUrl = 'file://{0}/{1}'.format(os.getcwd(), mapFname)
    #webbrowser.open_new(r"D:/Research Dr. Stanley (MSU)/GUI for Last-mile logistics system/GUI for last-mile logistics system/route_map_small.html")         

    driver = webdriver.Chrome()
    driver.get(mapUrl)
    # wait for 5 seconds for the maps and other assets to be loaded in the browser
    time.sleep(10)
    driver.save_screenshot('route_map_small.png')
    driver.quit()
    image_open6 = Image.open('route_map_small.png')
    image_resize6 = image_open6.resize((310,290),Image.ANTIALIAS)
    image_resize6.save('route_map_small_resize.png')
    image6 = PhotoImage(file='route_map_small_resize.png')
    image6.subsample(1,1)
    show_image6 = Canvas(right_frame1,width=310, height=290)
    show_image6.grid(row=1,column=0, columnspan=2, padx=5, pady=5)
    show_image6.place(relx=0.7, rely=0)
    show_image6.create_image(0,0, image=image6, anchor="nw")
    show_image6.image = image6  
    
    def plot_show6():
        webbrowser.open_new(r"D:/Research Dr. Stanley (MSU)/GUI for Last-mile logistics system/GUI for last-mile logistics system/route_map_small.html")         
        plt.close()
    
    plot_button6 = Button(right_frame1, text="Zoom",command =plot_show6, font=('times', 12, 'bold'))
    plot_button6.grid(row=2, column=0, padx=5, pady=5)
    plot_button6.place(relx=0.7, rely=0.9)
    
    def save_data6():
        route_perf_df = pd.DataFrame()
        route_perf_df["customerid"] = data_day["customerid"]
        route_perf_df["lat"] = data_day["lat"]
        route_perf_df["long"] = data_day["long"]
        route_perf_df["route"] = data_day["route"]
        #route_perf_df.to_csv("route_perf_small_scale.csv", index=False)
        try:
            with filedialog.asksaveasfile(mode='w', defaultextension=".csv") as file:
                route_perf_df.to_csv(file.name)
        except:
           print()
       
    down_button6 = Button(right_frame1, text="Save data",command = save_data6, font=('times', 12, 'bold'))
    down_button6.grid(row=2, column=0, padx=5, pady=5)
    down_button6.place(relx=0.77, rely=0.9)
    
    
    def destroy_6():
        show_image6.destroy()
        plot_button6.destroy()
        destroy_button6.destroy()
        down_button6.destroy()
        
    destroy_button6 = Button(right_frame1, text="Exit",command =destroy_6, font=('times', 12, 'bold'))
    destroy_button6.grid(row=2, column=0, padx=5, pady=5)
    destroy_button6.place(relx=0.87, rely=0.9)

        
totalcost_button = Button(left_frame2, text="Total Cost", command=CVRP_NEW, font=('times', 12, 'bold'),width =14)
totalcost_button.grid(row=4, column=0, padx=5, pady=5,sticky=W)

    
# Result message in left_frame3

Label_6 = Label(left_frame3, text="Small Scale",relief=RAISED, fg="white",bg="black",font=('times', 12, 'bold'),width =20)
Label_6.grid(row=1, column=1,columnspan=3, pady=10)

Label_7 = Label(left_frame3, text="Optimal",relief=RAISED, fg="white",bg="black",font=('times', 12, 'bold'),width =7)
Label_7.grid(row=2, column=1, pady=10, sticky=W)

Label_8 = Label(left_frame3, text="Baseline",relief=RAISED, fg="white",bg="black",font=('times', 12, 'bold'),width =7)
Label_8.grid(row=2, column=2, pady=10, sticky=W)

Label_9 = Label(left_frame3, text="Differ",relief=RAISED, fg="white",bg="black",font=('times', 12, 'bold'),width =7)
Label_9.grid(row=2, column=3, pady=10, sticky=W)
#--------------------------------------

#----------------Optimal
show_total_distance_best_success_objective_small = Entry(left_frame3, font=('times',12,'bold'), width =7)
show_total_distance_best_success_objective_small.grid(row=3,column=1, padx=5, pady=5)

show_total_fail_cost_best_success_objective_small = Entry(left_frame3,font=('times',12,'bold'), width =7)
show_total_fail_cost_best_success_objective_small.grid(row=4,column=1, padx=5, pady=5)


show_total_cost_best_success_objective_small= Entry(left_frame3,font=('times',12,'bold'), width =7)
show_total_cost_best_success_objective_small.grid(row=5,column=1, padx=5, pady=5)

#-----------------Baseline
show_round_total_distance_actual_small= Entry(left_frame3, font=('times',12,'bold'), width =7)
show_round_total_distance_actual_small.grid(row=3,column=2, padx=5, pady=5)

show_round_total_fail_cost_actual_small = Entry(left_frame3,font=('times',12,'bold'), width =7)
show_round_total_fail_cost_actual_small.grid(row=4,column=2, padx=5, pady=5)

show_round_total_cost_actual_small = Entry(left_frame3,font=('times',12,'bold'), width =7)
show_round_total_cost_actual_small.grid(row=5,column=2, padx=5, pady=5)

#-----------------Differ show
show_diff_total_distance_cost_small = Entry(left_frame3, font=('times',12,'bold'), width =7)
show_diff_total_distance_cost_small.grid(row=3,column=3, padx=5, pady=5)

show_diff_total_fail_cost_small = Entry(left_frame3,font=('times',12,'bold'), width =7)
show_diff_total_fail_cost_small.grid(row=4,column=3, padx=5, pady=5)


show_diff_total_cost_small = Entry(left_frame3,font=('times',12,'bold'), width =7)
show_diff_total_cost_small.grid(row=5,column=3, padx=5, pady=5)


#----------------------large-scale problem 

#------------------Optimal workload 
def Optimal_capacity_large():
    data = pd.read_csv(r"D:\Research Dr. Stanley (MSU)\GUI for Last-mile logistics system\GUI for last-mile logistics system\data_out_modified_40drop_own_final30.csv")
    data = data[data['attempt_date']<='2015-10-01']
    data.fillna(-1, inplace=True)
    data.sort_values(by=["route", "route_rank"], inplace=True)
    data_route = data["route"].values
    data_route_rank = data["route_rank"].values
    data_gc_dist = data["gc_dist"].values
    data_lat = data["lat"].values
    data_long = data["long"].values
    great_dist_diff = np.zeros(len(data))
    data["great_dist_diff"] = great_dist_diff
    data["log_greatdistdiff"] = np.log1p(data["great_dist_diff"])
    data["attempthour"] = (data["attempthour"].str[:2].astype(int)/6).astype(int)
    # New customerid_day_frequency 
    customer_days_dict = {}
    customer_id = data["customerid"].values
    attempt_date = data["attempt_date"].values
    for n in range(0, len(data)):
        if customer_id[n] not in customer_days_dict:
            customer_days_dict[customer_id[n]] = set()
        customer_days_dict[customer_id[n]].add(attempt_date[n])
    for customer in customer_days_dict:
        customer_days_dict[customer] = np.array(list(customer_days_dict[customer]))

    customerid_day_frequency_new = np.zeros(len(data))
    for n in range(0, len(data)):
        customerid_day_frequency_new[n] = np.sum(customer_days_dict[customer_id[n]] <= attempt_date[n])
    data["customerid_day_frequency_new"] = customerid_day_frequency_new
    
    # New zipcode_count
    zipcode_dict = {}
    zipcode = data["zipcode"].values
    attempt_date = data["attempt_date"].values
    for n in range(0, len(data)):
        if zipcode[n] not in zipcode_dict:
            zipcode_dict[zipcode[n]] = []
        zipcode_dict[zipcode[n]].append(attempt_date[n])
    for zipc in zipcode_dict:
        zipcode_dict[zipc] = np.array(zipcode_dict[zipc])

    zipcode_count_new = np.zeros(len(data))
    for n in range(0, len(data)):
        zipcode_count_new[n] = np.sum(zipcode_dict[zipcode[n]] <= attempt_date[n])
    data["zipcode_count_new"] = zipcode_count_new
    days = sorted(list(set(data["attempt_date"])))
    
    # Simulate days to compute optimization performance 
    for day in days:
        if day < "2015-02-02":
            continue
        data_day = data[data["attempt_date"]==day].copy(deep=True)
        routes = set(data_day["route"].tolist()) 
        
        for route in routes:
            data_route = data_day[data_day["route"]==route].copy(deep=True)
            data_route.sort_values(by="route_rank", inplace=True, ignore_index=True)  
            drops_in_route = len(data_route)
        
        Optimal_work_large =  round((drops_in_route*2-math.log(max(data_route_rank))),2)
        
    show_OptimapCapacity2.delete(0, 'end')
    show_OptimapCapacity2.insert(0, str( Optimal_work_large))
  
#Result message in left_frame1

Optimal_capacity_button = Button(left_frame2, command = Optimal_capacity_large,text="Optimal Workload",font=('times', 12, 'bold'),width =14)
Optimal_capacity_button.grid(row=2, column=1, padx=5, pady=5, stick = W)
show_OptimapCapacity2 = Entry(left_frame1, font=('times',12,'bold'), width =7)
show_OptimapCapacity2.grid(row=5,column=2, padx=5, pady=5)


#---------------------Failure Probability 

def Failure_Probability_large():
    from os.path import exists 
    for s_thresh in range(900,1000,50000):
        route_id_count = 0
        s_thresh /= 1000 
        
    if file_name:
        data = pd.read_csv(file_name)
    else:
        data = pd.read_csv(r"D:\Research Dr. Stanley (MSU)\GUI for Last-mile logistics system\GUI for last-mile logistics system\data_out_modified_40drop_own_final30.csv")    
    data = data[data['attempt_date']<='2015-10-01']
    data.fillna(-1, inplace=True)
    data.sort_values(by=["route", "route_rank"], inplace=True)

    # Construct some features used in the models 
    data["attempthour"] = (data["attempthour"].str[:2].astype(int)/6).astype(int)

    # New customerid_day_frequency 
    customer_days_dict = {}
    customer_id = data["customerid"].values
    attempt_date = data["attempt_date"].values
    for n in range(0, len(data)):
        if customer_id[n] not in customer_days_dict:
            customer_days_dict[customer_id[n]] = set()
        customer_days_dict[customer_id[n]].add(attempt_date[n])
    for customer in customer_days_dict:
        customer_days_dict[customer] = np.array(list(customer_days_dict[customer]))

    customerid_day_frequency_new = np.zeros(len(data))
    for n in range(0, len(data)):
        customerid_day_frequency_new[n] = np.sum(customer_days_dict[customer_id[n]] <= attempt_date[n])
    data["customerid_day_frequency_new"] = customerid_day_frequency_new

    # New zipcode_count
    zipcode_dict = {}
    zipcode = data["zipcode"].values
    attempt_date = data["attempt_date"].values
    for n in range(0, len(data)):
        if zipcode[n] not in zipcode_dict:
            zipcode_dict[zipcode[n]] = []
        zipcode_dict[zipcode[n]].append(attempt_date[n])
    for zipc in zipcode_dict:
        zipcode_dict[zipc] = np.array(zipcode_dict[zipc])

    zipcode_count_new = np.zeros(len(data))
    for n in range(0, len(data)):
        zipcode_count_new[n] = np.sum(zipcode_dict[zipcode[n]] <= attempt_date[n])
    data["zipcode_count_new"] = zipcode_count_new

    use_features = ["own_other", "route_rank", "shipments_route", "gc_dist"]
    use_features += ["shipmentamount", "shipmentweight", "Average_Temperature", "crime_month_count", "hh_size", "avg_income", "pop_densit"]
    use_features += ["att_weekday"]
    use_features += ["customerid"]
    use_features += ["customerid_day_frequency_new", "customer_day_packages", "failure_rate_date_lag1", "failure_rate_date_ma_week1", "deliveries_date_ma_week1", "days_since_last_delivery", "days_since_last_delivery_failure"]
    use_features += ["zipcode", "zipcode_count_new", "zipcode_day_count", "failure_rate_date_zipcode_lag1", "day_zipcode_frequency", "retail_dens", "literate_hoh"]
    use_features += ["lat_diff_abs", "long_diff_abs"]
    use_features += ["att_mnth", "dropsize"]

    data["lat_diff_abs"].fillna(-1, inplace=True)
    data["long_diff_abs"].fillna(-1, inplace=True)

    days = sorted(list(set(data["attempt_date"])))


    # Simulate days to compute optimization performance 
    for day in days:
        if day < "2015-10-01":
            continue
        data_day = data[data["attempt_date"]==day].copy(deep=True)
       
        # Select model
        model_type = "lgb"
        
        if model_type == "lgb":
            depth = 2
            feat_frac=0.3
            subsample_frac =0.6
            num_iter = 200
            if not exists ('gbm_model_save.txt'):
                continue
            gbm = lgb.Booster(model_file='gbm_model_save.txt')
        
        if model_type =="lgb":
            data_day["pred_failure"] = gbm.predict(data_day[use_features], num_threads=1)
        else:
            data_day["pred_failure"] = gbm.predict_proba(data_day[use_features])[:, 1] 
            
        
        probabiltiy = round(data_day["pred_failure"]*100,2)
        ax = sns.displot(probabiltiy, kde=True, bins=45)
        import matplotlib.ticker as mtick
        fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
        xticks = mtick.FormatStrFormatter(fmt)
        plt.gca().xaxis.set_major_formatter(xticks)
        ax.set(xlabel='Failure probability',
               ylabel='Count')
        plt.legend(title='Failure probability distribution', title_fontsize='large')
        plt.savefig('pred_failure.png')
        image_open5 = Image.open('pred_failure.png')
        image_resize5 = image_open5.resize((310,290), Image.ANTIALIAS)
        image_resize5.save('pred_failure_resize.png')
        image5= PhotoImage(file='pred_failure_resize.png')
        image5.subsample(1,1)
        show_image5 = Canvas(right_frame2,width=310, height=290)
        show_image5.grid(row=1,column=0, padx=5, pady=5)
        show_image5.place(relx=0, rely=0)
        show_image5.create_image(0,0, image=image5, anchor="nw")
        show_image5.image = image5
        
    def plot_show5():
        plt.show()
        plt.close()
        
    plot_button5 = Button(right_frame2, text="Zoom",command =plot_show5, font=('times', 12, 'bold'))
    plot_button5.grid(row=2, column=0, padx=5, pady=5)
    plot_button5.place(relx=0, rely=0.9)


    def save_data5():
        probability_data = pd.DataFrame()
        probability_data["customerid"] = data_day["customerid"]
        probability_data["pred_failure"] = data_day["pred_failure"]
        probability_data["lat"] = data_day["lat"]
        probability_data["long"] = data_day["long"]
        try:
            with filedialog.asksaveasfile(mode='w', defaultextension=".csv") as file:
                probability_data.to_csv(file.name)
        except:
           print()
       
    down_button5 = Button(right_frame2, text="Save data",command = save_data5, font=('times', 12, 'bold'))
    down_button5.grid(row=2, column=0, padx=5, pady=5)
    down_button5.place(relx=0.07, rely=0.9)
        


    def destroy_5():
        
        show_image5.destroy()
        plot_button5.destroy()
        destroy_button5.destroy()
        down_button5.destroy()
        
    destroy_button5 = Button(right_frame2, text="Exit",command =destroy_5, font=('times', 12, 'bold'))
    destroy_button5.grid(row=2, column=0, padx=5, pady=5)
    destroy_button5.place(relx=0.17, rely=0.9)
    
      
prob_large_button = Button(left_frame2, text="Failure Probability",font=('times', 12, 'bold'), command =Failure_Probability_large, width =14)
prob_large_button.grid(row=3, column=1, padx=5, pady=5, stick = W)     
      
       
def CVRP_large_scale(): 
    from tqdm.tk import tqdm
    from time import sleep
    pbar = tqdm(tk_parent=root)  # Create the progress bar ahead of time
    pbar._tk_window.withdraw()  # Hide it immediately
    pbar._tk_window.deiconify()
    pbar.reset(total=30)
    for _ in range(30):
        sleep(0.1)
        pbar.update(1)
    pbar._tk_window.withdraw()

    def min_objective_greedy_with_return_with_random_euc(OC_lat_pos, OC_long_pos, route_data):
        
        route_drops = []
        route_dist = []
        route_set = set()

        base_pred_success_mat = np.zeros((len(route_data), len(route_data)))
        base_pred_dist_mat = np.zeros((len(route_data), len(route_data)))
        pos_success_dict = {}
        pos_dist_dict = {}
        for pos in route_data["route_rank"].astype(int).values:
            base_df = route_data.copy(deep=True)
            base_df["route_rank"] = pos
            if len(route_data) > 1:
                pairwise_lat = np.array(list(combinations(route_data["lat"].tolist(), 2)))
                pairwise_long = np.array(list(combinations(route_data["long"].tolist(), 2)))
                avg_pairwise_lat_diff = np.average(np.abs(pairwise_lat[:, 0] - pairwise_lat[:, 1]))
                avg_pairwise_long_diff = np.average(np.abs(pairwise_long[:, 0] - pairwise_long[:, 1]))
            else:
                avg_pairwise_lat_diff = 0
                avg_pairwise_long_diff = 0
            base_df["lat_diff_abs"] = avg_pairwise_lat_diff
            base_df["long_diff_abs"] = avg_pairwise_long_diff
            if model_type=="lgb":
                base_pred_success = 1 - gbm.predict(base_df[use_features], num_threads=1)
            else:
                base_pred_success = 1 - gbm.predict_proba(base_df[use_features])[:, 1]
            base_pred_dist = np.array(base_df["lat_diff_abs"]*DIST_COST_LAT + base_df["long_diff_abs"]*DIST_COST_LONG)
            base_pred_success_mat[:, pos-1] = base_pred_success
            base_pred_dist_mat[:, pos-1] = np.array(base_pred_dist)
            pos_success_dict[pos] = base_pred_success
            pos_dist_dict[pos] = base_pred_dist
        base_pred_success = 0*base_pred_success
        base_pred_dist = 0*base_pred_dist
        for pos in pos_success_dict:
            base_pred_success += pos_success_dict[pos]
            base_pred_dist += pos_dist_dict[pos]
        base_pred_success /= len(pos_success_dict)
        base_pred_dist /= len(pos_dist_dict)
        base_pred_success = np.mean(base_pred_success_mat, axis=1)
        base_pred_dist = np.mean(base_pred_dist_mat, axis=1)
            
        lat_vals = route_data["lat"].values
        long_vals = route_data["long"].values
         
        while len(route_drops) < len(lat_vals):
            min_obj = 1000000000000
            max_node = None
            SUCCESS_WEIGHT_START=1
            SUCCESS_WEIGHT_MID=1
            if len(route_drops) == 0:                
                route_data["route_rank"] = 1
                route_data["lat_diff_abs"] = np.abs(OC_lat_pos - route_data["lat"])
                route_data["long_diff_abs"] = np.abs(OC_long_pos - route_data["long"])
                route_data["log_greatdistdiff"] = route_data.apply(lambda x: np.log1p(great_distance(OC_lat_pos, OC_long_pos, x["lat"], x["long"])), axis=1) # Great distance
                if model_type=="lgb":
                    pred_success = 1 - gbm.predict(route_data[use_features], num_threads=1)
                else:
                    pred_success = 1 - gbm.predict_proba(route_data[use_features])[:, 1]
                pred_success_diff = base_pred_success - pred_success
                dist_diff = np.abs(OC_lat_pos - route_data["lat"])*DIST_COST_LAT + np.abs(OC_long_pos - route_data["long"])*DIST_COST_LONG
                obj_vals = dist_diff + (SUCCESS_WEIGHT_START*pred_success_diff) * FAIL_COST
                for n in range(0, len(pred_success_diff)):
                    if n not in route_set:
                        if obj_vals[n] < min_obj:
                            min_obj = obj_vals[n]
                            max_node = n
                route_set.add(max_node)
                route_drops.append(max_node)
                route_dist.append(great_distance(OC_lat_pos, OC_long_pos, lat_vals[max_node], long_vals[max_node])) # Great distance
                prev_lat = lat_vals[max_node]
                prev_long = long_vals[max_node]
            else:
                route_data["route_rank"] = len(route_drops) + 1
                route_data["lat_diff_abs"] = np.abs(prev_lat - route_data["lat"])
                route_data["long_diff_abs"] = np.abs(prev_long - route_data["long"])
                route_data["log_greatdistdiff"] = route_data.apply(lambda x: np.log1p(great_distance(prev_lat, prev_long, x["lat"], x["long"])), axis=1) # Great distance
                if model_type=="lgb":
                    pred_success = 1 - gbm.predict(route_data[use_features], num_threads=1)
                else:
                    pred_success = 1 - gbm.predict_proba(route_data[use_features])[:, 1]
                pred_success_diff = base_pred_success - pred_success
                dist_diff = np.abs(prev_lat - route_data["lat"])*DIST_COST_LAT + np.abs(prev_long - route_data["long"])*DIST_COST_LONG          
                obj_vals = dist_diff + (SUCCESS_WEIGHT_MID*pred_success_diff) * FAIL_COST
                for n in range(0, len(pred_success_diff)):
                    if n not in route_set:
                        if obj_vals[n] < min_obj:
                            min_obj = obj_vals[n]
                            max_node = n
                route_set.add(max_node)
                route_drops.append(max_node)
                route_dist.append(great_distance(prev_lat, prev_long, lat_vals[max_node], long_vals[max_node])) # Great distance
                prev_lat = lat_vals[max_node]
                prev_long = long_vals[max_node]    
        
        route_dist.append(great_distance(OC_lat_pos, OC_long_pos, prev_lat, prev_long)*DIST_COST_GREAT) # Great distance
                
        rand_threshold = 0 # Driver deviation
        rand_nums = np.random.randint(1000, size=len(route_drops))
        route_dist = []
        
        for n in range(0, len(rand_nums)):
            if rand_nums[n] < rand_threshold and len(rand_nums) > 1:
                curr_drop = route_drops[n] 
                route_drops = route_drops[:n]+route_drops[n+1:]
                new_drop_pos = np.random.choice(route_drops, 1)[0]
                route_drops.insert(new_drop_pos, curr_drop)         
                
        route_dist.append(great_distance(OC_lat_pos, OC_long_pos, lat_vals[route_drops[0]], long_vals[route_drops[0]])*DIST_COST_GREAT) # Great distance
        for n in range(0, len(route_drops)-1):  
            route_dist.append(great_distance(lat_vals[route_drops[n]], long_vals[route_drops[n]], lat_vals[route_drops[n+1]], long_vals[route_drops[n+1]])*DIST_COST_GREAT) # Great distance
        
        route_dist.append(great_distance(OC_lat_pos, OC_long_pos, lat_vals[route_drops[-1]], long_vals[route_drops[-1]])*DIST_COST_GREAT) # Great distance
        route_seq_dict = {"route_drops":np.array(route_drops), "route_dist":np.array(route_dist)}
                         
        return route_seq_dict



    # Function to calculate total route (great) distance
    data = pd.read_csv(r"D:\Research Dr. Stanley (MSU)\GUI for Last-mile logistics system\GUI for last-mile logistics system\data_out_modified_40drop_own_final30.csv") 
    data = data[data['attempt_date']<='2015-10-01']
    def great_distance(x1, y1, x2, y2):
        
        if x1 == x2 and y1 == y2:
            return 0

        x1 = math.radians(x1)
        y1 = math.radians(y1)
        x2 = math.radians(x2)
        y2 = math.radians(y2)

        angle1 = math.acos(math.sin(x1) * math.sin(x2) + math.cos(x1) * math.cos(x2) * math.cos(y1 - y2))
        angle1 = math.degrees(angle1)

        return 60.0 * angle1

    # Calculate the great route distance for the existing dataset based on the original sequencing
    routedistance_dict = {}
    for route in set(data["route"].tolist()):
        route_vals = data[["route_rank", "lat", "long"]].loc[data["route"] == route].sort_values(by="route_rank").values
        total_dist = 0
        for n in range(len(route_vals)-1):
            total_dist += great_distance(route_vals[n, 1], route_vals[n, 2], route_vals[n+1, 1], route_vals[n+1, 2])
        routedistance_dict[route] = total_dist
        
    # Set parameters and simulate
    from os.path import exists 
    for s_thresh in range(900,1000,50000):
        route_id_list = []
        route_len_list = []
        route_dist_cost_actual_list = []
        route_dist_cost_best_objective_list = []
        route_failure_cost_actual_list = []
        route_failure_cost_best_objective_list = []
        route_total_cost_actual_list = []
        route_total_cost_best_objective_list = []
        

        route_id_count = 0
        s_thresh /= 1000 
        
        data = pd.read_csv(r"D:\Research Dr. Stanley (MSU)\GUI for Last-mile logistics system\GUI for last-mile logistics system\data_out_modified_40drop_own_final30.csv") 
        data = data[data['attempt_date']<='2015-10-01']
        data.fillna(-1, inplace=True)
        data.sort_values(by=["route", "route_rank"], inplace=True)
        
        # Construct some features used in the models 
        routedistance_dict = {}
        for route in set(data["route"].tolist()):
            route_vals = data[["route_rank", "lat", "long"]].loc[data["route"] == route].sort_values(by="route_rank").values
            total_dist = 0
            for n in range(len(route_vals)-1):
                total_dist += great_distance(route_vals[n, 1], route_vals[n, 2], route_vals[n+1, 1], route_vals[n+1, 2])
            routedistance_dict[route] = total_dist

        data["totalroutedistance"] = data["route"].apply(lambda x: routedistance_dict[x])

        data["log_gc_dist"] = np.log1p(data["gc_dist"])
        data["log_shipmentamount"] = np.log1p(data["shipmentamount"])
        data["log_shipmentweight"] = np.log1p(data["shipmentweight"])
        data["log_totalroutedistance"] = np.log1p(data["totalroutedistance"])

        data_route = data["route"].values
        data_route_rank = data["route_rank"].values
        data_gc_dist = data["gc_dist"].values
        data_lat = data["lat"].values
        data_long = data["long"].values
        great_dist_diff = np.zeros(len(data))
        for n in range(0, len(data)):
            if n == 0 or data_route[n] != data_route[n-1]:
                great_dist_diff[n] = data_gc_dist[n]
            else:
                great_dist_diff[n] = great_distance(data_lat[n], data_long[n], data_lat[n-1], data_long[n-1])
        data["great_dist_diff"] = great_dist_diff
        data["log_greatdistdiff"] = np.log1p(data["great_dist_diff"])
        data["attempthour"] = (data["attempthour"].str[:2].astype(int)/6).astype(int)
        
        # New customerid_day_frequency 
        customer_days_dict = {}
        customer_id = data["customerid"].values
        attempt_date = data["attempt_date"].values
        for n in range(0, len(data)):
            if customer_id[n] not in customer_days_dict:
                customer_days_dict[customer_id[n]] = set()
            customer_days_dict[customer_id[n]].add(attempt_date[n])
        for customer in customer_days_dict:
            customer_days_dict[customer] = np.array(list(customer_days_dict[customer]))

        customerid_day_frequency_new = np.zeros(len(data))
        for n in range(0, len(data)):
            customerid_day_frequency_new[n] = np.sum(customer_days_dict[customer_id[n]] <= attempt_date[n])
        data["customerid_day_frequency_new"] = customerid_day_frequency_new
        
        # New zipcode_count
        zipcode_dict = {}
        zipcode = data["zipcode"].values
        attempt_date = data["attempt_date"].values
        for n in range(0, len(data)):
            if zipcode[n] not in zipcode_dict:
                zipcode_dict[zipcode[n]] = []
            zipcode_dict[zipcode[n]].append(attempt_date[n])
        for zipc in zipcode_dict:
            zipcode_dict[zipc] = np.array(zipcode_dict[zipc])

        zipcode_count_new = np.zeros(len(data))
        for n in range(0, len(data)):
            zipcode_count_new[n] = np.sum(zipcode_dict[zipcode[n]] <= attempt_date[n])
        data["zipcode_count_new"] = zipcode_count_new
        
        use_features = ["own_other", "route_rank", "shipments_route", "gc_dist"]
        use_features += ["shipmentamount", "shipmentweight", "Average_Temperature", "crime_month_count", "hh_size", "avg_income", "pop_densit"]
        use_features += ["att_weekday"]
        use_features += ["customerid"]
        use_features += ["customerid_day_frequency_new", "customer_day_packages", "failure_rate_date_lag1", "failure_rate_date_ma_week1", "deliveries_date_ma_week1", "days_since_last_delivery", "days_since_last_delivery_failure"]
        use_features += ["zipcode", "zipcode_count_new", "zipcode_day_count", "failure_rate_date_zipcode_lag1", "day_zipcode_frequency", "retail_dens", "literate_hoh"]
        use_features += ["lat_diff_abs", "long_diff_abs"]
        use_features += ["att_mnth", "dropsize"]

        data["lat_diff_abs"].fillna(-1, inplace=True)
        data["long_diff_abs"].fillna(-1, inplace=True)
        
        # Compute distance from each drop to OC
        OC_lat = np.array([-23.558123100, -23.433411500, -23.558123100, -23.514664500, -23.526479600, -23.661295500, -23.671398600, -23.492313700])
        OC_long = np.array([-46.609302200, -46.554093100, -46.609302200, -46.653909400, -46.766178900, -46.487164700, -46.716471400, -46.842962500])
        OC_dict = {}
        for OC in set(data["OC"].tolist()):
            data_OC = data[["lat", "long"]].loc[data["OC"]==OC]
            for n in range(0, len(OC_lat)):
                lat_dist = np.abs(data_OC["lat"] - OC_lat[n])
                long_dist = np.abs(data_OC["long"] - OC_long[n])

        OC_dict = {"OC01":3, "OC02":5, "OC03":1, "OC04":0, "OC05":6, "OC06":4, "OC10":2, "OC09":7}
        data["lat_diff_OC"] = data["OC"].apply(lambda x: OC_lat[OC_dict[x]])
        data["long_diff_OC"] = data["OC"].apply(lambda x: OC_long[OC_dict[x]])
        
        # Parameters for travel cost and failure cost 
        TRAVEL_COST_MULTIPLIER = 1
        TRAVEL_SPEED = CapacityofVehicle2.get()
        DRIVER_WAGE = 1
        DIST_COST_LAT = (0.58/15/1.61 + DRIVER_WAGE/TRAVEL_SPEED)*110.57*TRAVEL_COST_MULTIPLIER
        DIST_COST_LONG = (0.58/15/1.61 + DRIVER_WAGE/TRAVEL_SPEED)*102.05*TRAVEL_COST_MULTIPLIER
        DIST_COST_GREAT = (0.58/15/1.61 + DRIVER_WAGE/TRAVEL_SPEED)/1.852*TRAVEL_COST_MULTIPLIER
        FAIL_COST = 10
        
        missed_thresholds = 0
        num_drops_total = 0
        
        total_distance_actual = 0
        total_distance_best_success_objective = 0
        
        success_actual_list = []
        success_best_success_objective_list = []
        
        total_cost_actual = 0
        total_cost_best_success_objective = 0
        
        days = sorted(list(set(data["attempt_date"])))
        use_vars = ["attempt_date", "route", "OC", "lat", "long"] + use_features
        weight_list = []
        
        # Simulate days to compute optimization performance 
        for day in days:
            if day < "2015-10-01":
                continue
            data_day = data[data["attempt_date"]==day].copy(deep=True)
            
            num_drops_day = 0
            total_distance_actual_day = 0
            total_distance_best_success_objective_day = 0
            success_actual_day_list = []
            success_best_success_objective_day_list = []
            total_cost_actual_day = 0
            total_cost_best_success_objective_day = 0
            
            
            # Select model
            model_type = "lgb"
            
            if model_type == "lgb":
                depth = 2
                feat_frac=0.3
                subsample_frac =0.6
                num_iter = 200
                if not exists ('gbm_model_save.txt'):
                    continue
                gbm = lgb.Booster(model_file='gbm_model_save.txt')
            
            if model_type =="lgb":
                data_day["pred_success"] = 1 - gbm.predict(data_day[use_features], num_threads=1)
            else:
                data_day["pred_success"] = 1 - gbm.predict_proba(data_day[use_features])[:, 1] 
            
            routes = set(data_day["route"].tolist()) 
            
            for route in routes:
                data_route = data_day[data_day["route"]==route].copy(deep=True)
                data_route.sort_values(by="route_rank", inplace=True, ignore_index=True)  
                OC_ind = data_route["OC"].values[0]
                average_route_success_actual = np.average(data_route["pred_success"])
                total_route_distance_actual = data_route["great_dist_diff"].sum() * DIST_COST_GREAT
                
                # Lowest objective greedy
                #drops_in_route = len(route_drops)
                best_success_route_objective_df = data_route.copy(deep=True)
                best_success_route_objective_dict = min_objective_greedy_with_return_with_random_euc(OC_lat[OC_dict[OC_ind]], OC_long[OC_dict[OC_ind]], best_success_route_objective_df)
                best_success_route_objective_sequence = best_success_route_objective_dict["route_drops"]
                new_route_sequence = np.zeros(len(best_success_route_objective_df))
                for n in range(0, len(best_success_route_objective_sequence)):
                    new_route_sequence[best_success_route_objective_sequence[n]] = n+1
                best_success_route_objective_df["route_rank"] = new_route_sequence
                best_success_route_objective_df = best_success_route_objective_df.sort_values(by="route_rank", ignore_index=True)

                best_success_route_objective_df["lat_diff"] = best_success_route_objective_df["lat"].diff()
                best_success_route_objective_df["long_diff"] = best_success_route_objective_df["long"].diff()
                best_success_route_objective_df["lat_diff_abs"] = np.abs(best_success_route_objective_df["lat_diff"])
                best_success_route_objective_df["long_diff_abs"] = np.abs(best_success_route_objective_df["long_diff"])

                best_success_route_objective_df["lat_diff_abs"].fillna(-1, inplace=True)
                best_success_route_objective_df["long_diff_abs"].fillna(-1, inplace=True)

                best_success_route_objective_df["log_greatdistdiff"] = best_success_route_objective_df.apply(lambda x: np.log1p(great_distance(x["lat"], x["long"], x["lat"]+x["lat_diff"], x["long"]+x["long_diff"])) 
                                                                                  if x["long_diff_abs"]!=-1 else np.log1p(x["gc_dist"]), axis=1) # Great distance
                if model_type =="lgb":
                    best_success_route_objective_success = 1 - gbm.predict(best_success_route_objective_df[use_features], num_threads=1)
                else:
                    best_success_route_objective_success = 1 - gbm.predict_proba(best_success_route_objective_df[use_features])[:, 1]
                best_success_route_objective_df["pred_success"] = best_success_route_objective_success
                average_best_success_route_objective_success = np.average(best_success_route_objective_success)
                total_best_success_route_objective_distance = np.sum(best_success_route_objective_dict["route_dist"])   
                
                # Only consider routes with at least 1 drop 
                
                drops_in_route = len(data_route)
                if drops_in_route > 0:
                        num_drops_day += len(data_route)
                        total_distance_actual_day += total_route_distance_actual
                        total_distance_best_success_objective_day += total_best_success_route_objective_distance
                        success_actual_day_list += data_route["pred_success"].tolist()
                        success_best_success_objective_day_list += best_success_route_objective_df["pred_success"].tolist()
                        total_cost_actual_day += total_route_distance_actual + np.sum(1 - data_route["pred_success"].values) * FAIL_COST
                        total_cost_best_success_objective_day += total_best_success_route_objective_distance + np.sum(1 - best_success_route_objective_df["pred_success"].values) * FAIL_COST
                    
                route_id_count += 1
                route_id_list.append(route_id_count)
                route_len_list.append(len(data_route))
                
                route_dist_cost_actual_list.append(total_route_distance_actual)
                route_dist_cost_best_objective_list.append(total_best_success_route_objective_distance)
                
                route_failure_cost_actual_list.append(np.sum(1 - data_route["pred_success"].values) * FAIL_COST)
                route_failure_cost_best_objective_list.append(np.sum(1 - best_success_route_objective_df["pred_success"].values) * FAIL_COST)
                
                route_total_cost_actual_list.append(total_route_distance_actual+ np.sum(1 - data_route["pred_success"].values) * FAIL_COST)
                route_total_cost_best_objective_list.append(total_best_success_route_objective_distance + np.sum(1 - best_success_route_objective_df["pred_success"].values) * FAIL_COST)

            num_drops_total += num_drops_day
            total_distance_actual += total_distance_actual_day
            total_distance_best_success_objective += total_distance_best_success_objective_day
            success_actual_list += success_actual_day_list
            success_best_success_objective_list += success_best_success_objective_day_list
            total_cost_actual += total_cost_actual_day
            total_cost_best_success_objective += total_cost_best_success_objective_day  

                
               
                
        # print(day, num_drops_day, np.average(success_actual_day_list), np.average(success_shortest_day_list), np.average(success_best_success_day_list), np.average(success_best_success_objective_day_list), np.average(success_best_success_objective_day_list_with_success_threshold), np.average(success_random_route_day_list), total_distance_actual_day, total_distance_shortest_day, total_distance_best_success_day, total_distance_best_success_objective_day, total_distance_best_success_objective_day_with_success_threshold, total_distance_random_route_day, total_cost_actual_day, total_cost_shortest_day, total_cost_best_success_day, total_cost_best_success_objective_day, total_cost_best_success_objective_day_with_success_threshold, total_cost_random_route_day, success_threshold, missed_thresholds)

        #print("s_thresh", s_thresh, num_drops_total, total_distance_actual, 
              #total_distance_best_success_objective,  
              #FAIL_COST*num_drops_total*(1-np.average(success_actual_list)),
              #FAIL_COST*num_drops_total*(1-np.average(success_best_success_objective_list)), 
              #total_cost_actual,
              #total_cost_best_success_objective)
        
        round_total_distance_actual = round(total_distance_actual,2)
        #print(round_total_distance_actual)
        show_round_total_distance_actual.delete(0, 'end')
        show_round_total_distance_actual.insert(0, str("$")+str(round_total_distance_actual))
       
        
        round_total_distance_best_success_objective = round(total_distance_best_success_objective,2)
        #print(round_total_distance_best_success_objective )
        show_total_distance_best_success_objective.delete(0, 'end')
        show_total_distance_best_success_objective.insert(0, str("$")+str(round_total_distance_best_success_objective))
       
        

        
        total_fail_cost_actual = FAIL_COST*num_drops_total*(1-np.average(success_actual_list))
        round_total_fail_cost_actual = round( total_fail_cost_actual,2)
        #print( round_total_fail_cost_actual)
        show_round_total_fail_cost_actual.delete(0, 'end')
        show_round_total_fail_cost_actual.insert(0, str("$")+str(round_total_fail_cost_actual))
       
        
        
        total_fail_cost_best_success_objective = FAIL_COST*num_drops_total*(1-np.average(success_best_success_objective_list))
        round_total_fail_cost_best_success_objective  = round(total_fail_cost_best_success_objective,2)
        #print(round_total_fail_cost_best_success_objective)
        show_total_fail_cost_best_success_objective.delete(0, 'end')
        show_total_fail_cost_best_success_objective.insert(0, str("$")+str(round_total_fail_cost_best_success_objective))
        
        
        round_total_cost_actual = round(total_cost_actual,2)
        #print(round_total_cost_actual)
        show_round_total_cost_actual.delete(0, 'end')
        show_round_total_cost_actual.insert(0, str("$")+str( round_total_cost_actual))
        
        
        round_total_cost_best_success_objective = round(total_cost_best_success_objective,2)
        #print(round_total_cost_best_success_objective)
        show_total_cost_best_success_objective.delete(0, 'end')
        show_total_cost_best_success_objective.insert(0, str("$")+str(round_total_cost_best_success_objective))
        
        # Difference
        diff_total_distance_cost = round(((round_total_distance_best_success_objective-round_total_distance_actual)/round_total_distance_actual)*100,2)
        show_diff_total_distance_cost.delete(0, 'end')
        show_diff_total_distance_cost.insert(0, str( diff_total_distance_cost)+str("%"))
        
        diff_total_fail_cost = round(((round_total_fail_cost_best_success_objective-round_total_fail_cost_actual)/round_total_fail_cost_actual)*100,2)
        show_diff_total_fail_cost.delete(0, 'end')
        show_diff_total_fail_cost.insert(0, str( diff_total_fail_cost)+str("%"))
        
        diff_total_cost = round(((round_total_cost_best_success_objective-round_total_cost_actual)/round_total_cost_actual)*100,2)
        show_diff_total_cost.delete(0, 'end')
        show_diff_total_cost.insert(0, str(diff_total_cost)+str("%"))
    
        
        #--------------------Bar gragh
        X_set = ['Travel Cost', 'Failure Cost', 'Total Cost']
        Y_actual = [round_total_distance_actual, round_total_fail_cost_actual, round_total_cost_actual]
        Y_best = [round_total_distance_best_success_objective, round_total_fail_cost_best_success_objective, round_total_cost_best_success_objective]
        
        X_axis = np.arange(len(X_set))
        
        plt.bar(X_axis - 0.2, Y_actual, 0.4, label = 'Baseline')
        for i in range(len(X_axis)):
            plt.text(i,Y_actual[i],Y_actual[i], ha='right',color='green', fontweight='bold')
        
        plt.bar(X_axis + 0.2, Y_best, 0.4, label = 'Optimal')
        for i in range(len(X_axis)):
            plt.text(i,Y_best[i],Y_best[i], ha='left',color='red', fontweight='bold')
        
        plt.xticks(X_axis, X_set)
        plt.xlabel("Cost groups")
        plt.ylabel("Cost ($)")
        plt.title("Costs comparison of CVRP model with/without failure probability")
        plt.legend()
        plt.savefig('Large_scale_comparison.png')
        
        #------------Open saved pic
        image_open4 = Image.open('Large_scale_comparison.png')
        image_resize4 = image_open4.resize((310,290),Image.ANTIALIAS)
        image_resize4.save('Large_scale_comparison_resize.png')
        image4 = PhotoImage(file="Large_scale_comparison_resize.png")
        image4.subsample(1,1)
        show_image4 = Canvas(right_frame2,width=310, height=290)
        show_image4.grid(row=1,column=0, columnspan=2, padx=5, pady=5)
        show_image4.place(relx=0.35, rely=0)
        show_image4.create_image(0,0, image=image4, anchor="nw")
        show_image4.image = image4
        
       
        
        def plot_show4():
            plt.show()
            plt.close()
        plot_button4 = Button(right_frame2, text="Zoom",command =plot_show4, font=('times', 12, 'bold'))
        plot_button4.grid(row=2, column=0, padx=5, pady=5)
        plot_button4.place(relx=0.35, rely=0.9)
        
        
        def save_data4():
            cost_perf_df = pd.DataFrame()
            cost_perf_df["route_id"] = route_id_list
            cost_perf_df["route_len"] = route_len_list
            cost_perf_df["route_dist_cost_actual"] = route_dist_cost_actual_list
            cost_perf_df["route_dist_cost_best_objective"] = route_dist_cost_best_objective_list
            cost_perf_df["route_failure_cost_actual"] = route_failure_cost_actual_list
            cost_perf_df["route_failure_cost_best_objective"] = route_failure_cost_best_objective_list
            cost_perf_df["route_total_cost_actual"] = route_total_cost_actual_list
            cost_perf_df["route_total_cost_best_objective"] = route_total_cost_best_objective_list
            #cost_perf_df.to_csv("cost_perf_large.csv", index=False) 
            
            try:
                with filedialog.asksaveasfile(mode='w', defaultextension=".csv") as file:
                    cost_perf_df.to_csv(file.name)
            except:
               print()
               
            route_perf_large_df = pd.DataFrame()
            route_perf_large_df["customerid"] = data_day["customerid"]
            route_perf_large_df["lat"] = data_day["lat"]
            route_perf_large_df["long"] = data_day["long"]
            route_perf_large_df["route"] = data_day["route"]
            route_perf_large_df.to_csv("route_perf_large_scale.csv", index=False)
            
            
        down_button4 = Button(right_frame2, text="Save data",command = save_data4, font=('times', 12, 'bold'))
        down_button4.grid(row=2, column=0, padx=5, pady=5)
        down_button4.place(relx=0.42, rely=0.9)
             
        def destroy_4():
            show_image4.destroy()
            plot_button4.destroy()
            destroy_button4.destroy()
            down_button4.destroy()
        
        destroy_button4 = Button(right_frame2, text="Exit",command =destroy_4, font=('times', 12, 'bold'))
        destroy_button4.grid(row=2, column=0, padx=5, pady=5)
        destroy_button4.place(relx=0.52, rely=0.9)
        
        # Map route 
        def get_directions_response(lat1, long1, lat2, long2, mode='drive'):
            url = "https://route-and-directions.p.rapidapi.com/v1/routing"
            key = "165aea8a7fmsh8e6392177d6e3a5p1d9590jsn08d0be6112ff"
            host = "route-and-directions.p.rapidapi.com"
            headers = {"X-RapidAPI-Key": key, "X-RapidAPI-Host": host}
            querystring = {"waypoints":f"{str(lat1)},{str(long1)}|{str(lat2)},{str(long2)}","mode":mode}
            response = requests.request("GET", url, headers=headers, params=querystring)
            return response

        location = pd.read_csv(r"D:/Research Dr. Stanley (MSU)/GUI for Last-mile logistics system/GUI for last-mile logistics system/route_perf_large_scale.csv")

        location = location[["lat", "long"]]
        location_list = location.values.tolist()
        new_list = []
        for i in location_list:
           new_list.append(tuple(i))

        # Depot 
        df_depot = pd.DataFrame()
        depot_lat = [-23.514664500, -23.661295500, -23.433411500, -23.558123100, -23.671398600,-23.526479600,-23.492313700,-23.558123100]
        depot_long = [-46.653909400, -46.487164700, -46.554093100,-46.609302200, -46.716471400,-46.766178900, -46.842962500,-46.609302200]
        depot_name = ["OC01", "OC02", "OC03", "OC04", "OC05", "OC06", "OC09", "OC10"]
        df_depot['depot_name']= depot_name
        df_depot['depot_lat']= depot_lat
        df_depot['depot_long']= depot_long
        depot_location = df_depot[["depot_lat","depot_long"]]
        depot_location_list = depot_location.values.tolist()
        new_depot_list=[]
        for i in depot_location_list:
           new_depot_list.append(tuple(i))

        final_list =  new_list + new_depot_list

        m = folium.Map()
        colors = ['blue','red','green','black','maroon','orange', 'maroon', 'lime', 'teal', 'green']
        for point in new_list[:]:
            folium.Marker(point,icon=folium.Icon( color='blue')).add_to(m)
            
        for point in new_depot_list[:]:
            folium.Marker(point,icon=folium.Icon( color='red')).add_to(m)
           
        routes = [[213,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                   21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,213],
                  [214,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,
                   54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,214],
                  [215,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,215],
                  [216,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,216],
                  [217,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,217],
                  [217,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,
                   161,162,163,164,165,166,167,168,169,170,171,172,173,174,217],
                  [218,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,
                   197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,218]]
        route_trans = []
        for i in range(len(routes)):
            trans = []
            for shipment_index in routes[i]:
                trans.append(final_list[shipment_index]) 
            route_trans.append(trans) 
            
            
        responses = []
        for r in range(len( route_trans)):
            for n in range(len(route_trans[r])-1):
                lat1 = route_trans[r][n][0]
                lon1 = route_trans[r][n][1]
                lat2 = route_trans[r][n+1][0]
                lon2 = route_trans[r][n+1][1]
                response= get_directions_response(lat1, lon1, lat2, lon2, mode='drive')
                responses.append(response)
                mls = response.json()['features'][0]['geometry']['coordinates']
                points = [(i[1], i[0]) for i in mls[0]]
                folium.PolyLine(points, weight=5, opacity=1, color=colors[r]).add_to(m)
                temp = pd.DataFrame(mls[0]).rename(columns={0:'Lon', 1:'Lat'})[['Lat', 'Lon']]
                df = pd.DataFrame()
                df = pd.concat([df, temp])
                sw = df[['Lat', 'Lon']].min().values.tolist()
                sw = [sw[0]-0.0005, sw[1]-0.0005]
                ne = df[['Lat', 'Lon']].max().values.tolist()
                ne = [ne[0]+0.0005, ne[1]+0.0005]
                m.fit_bounds([sw, ne])   
                     
        # Save map
        mapFname = 'route_map_large.html'
        m.save(mapFname)
        mapUrl = 'file://{0}/{1}'.format(os.getcwd(), mapFname)
        #webbrowser.open_new(r"D:/Research Dr. Stanley (MSU)/GUI for Last-mile logistics system/GUI for last-mile logistics system/route_map_small.html")         

        driver = webdriver.Chrome()
        driver.get(mapUrl)
        # wait for 5 seconds for the maps and other assets to be loaded in the browser
        time.sleep(10)
        driver.save_screenshot('route_map_large.png')
        driver.quit()
        image_open7 = Image.open('route_map_large.png')
        image_resize7 = image_open7.resize((310,290),Image.ANTIALIAS)
        image_resize7.save('route_map_large_resize.png')
        image7 = PhotoImage(file='route_map_large_resize.png')
        image7.subsample(1,1)
        show_image7 = Canvas(right_frame2,width=310, height=290)
        show_image7.grid(row=1,column=0, columnspan=2, padx=5, pady=5)
        show_image7.place(relx=0.7, rely=0)
        show_image7.create_image(0,0, image=image7, anchor="nw")
        show_image7.image = image7  
        
        def plot_show7():
            webbrowser.open_new(r"D:/Research Dr. Stanley (MSU)/GUI for Last-mile logistics system/GUI for last-mile logistics system/route_map_large.html")         
            plt.close()
        
        plot_button7 = Button(right_frame2, text="Zoom",command =plot_show7, font=('times', 12, 'bold'))
        plot_button7.grid(row=2, column=0, padx=5, pady=5)
        plot_button7.place(relx=0.7, rely=0.9)
        
        def save_data7():
            route_perf_large_df = pd.DataFrame()
            route_perf_large_df["customerid"] = data_day["customerid"]
            route_perf_large_df["lat"] = data_day["lat"]
            route_perf_large_df["long"] = data_day["long"]
            route_perf_large_df["route"] = data_day["route"]
            route_perf_large_df.to_csv("route_perf_large_scale.csv", index=False)
            try:
                with filedialog.asksaveasfile(mode='w', defaultextension=".csv") as file:
                    route_perf_large_df.to_csv(file.name)
            except:
               print()
        down_button7 = Button(right_frame2, text="Save data",command = save_data7, font=('times', 12, 'bold'))
        down_button7.grid(row=2, column=0, padx=5, pady=5)
        down_button7.place(relx=0.77, rely=0.9) 
           
        
        def destroy_7():
            show_image7.destroy()
            plot_button7.destroy()
            destroy_button7.destroy()
            down_button7.destroy()
            
        destroy_button7 = Button(right_frame2, text="Exit",command =destroy_7, font=('times', 12, 'bold'))
        destroy_button7.grid(row=2, column=0, padx=5, pady=5)
        destroy_button7.place(relx=0.87, rely=0.9)
            

Large_scale_button = Button(left_frame2, command =CVRP_large_scale,  text="Total Cost", font=('times', 12, 'bold'), width =14)
Large_scale_button.grid(row=4,column = 1, padx=5, pady=5, stick = W)

Label_10 = Label(left_frame3, text="Large Scale",relief=RAISED, fg="white",bg="black",font=('times', 12, 'bold'), width =20)
Label_10.grid(row=1, column=4,columnspan=3, pady=10)

Label_11 = Label(left_frame3, text="Optimal",relief=RAISED, fg="white",bg="black",font=('times', 12, 'bold'),width =7)
Label_11.grid(row=2, column=4, pady=10, sticky=W)

Label_12 = Label(left_frame3, text="Baseline",relief=RAISED, fg="white",bg="black",font=('times', 12, 'bold'),width =7)
Label_12.grid(row=2, column=5, pady=10, sticky=W)

Label_13 = Label(left_frame3, text="Differ",relief=RAISED, fg="white",bg="black",font=('times', 12, 'bold'),width =7)
Label_13.grid(row=2, column=6, pady=10, sticky=W)


#----------------Optimal
show_round_total_distance_actual = Entry(left_frame3, font=('times',12,'bold'), width =7)
show_round_total_distance_actual.grid(row=3,column=4, padx=5, pady=5)

show_total_fail_cost_best_success_objective  = Entry(left_frame3,font=('times',12,'bold'), width =7)
show_total_fail_cost_best_success_objective.grid(row=4,column=4, padx=5, pady=5)


show_total_cost_best_success_objective = Entry(left_frame3,font=('times',12,'bold'), width =7)
show_total_cost_best_success_objective.grid(row=5,column=4, padx=5, pady=5)

#-----------------Baseline
show_total_distance_best_success_objective = Entry(left_frame3, font=('times',12,'bold'), width =7)
show_total_distance_best_success_objective.grid(row=3,column=5, padx=5, pady=5)

show_round_total_fail_cost_actual = Entry(left_frame3,font=('times',12,'bold'), width =7)
show_round_total_fail_cost_actual.grid(row=4,column=5, padx=5, pady=5)


show_round_total_cost_actual = Entry(left_frame3,font=('times',12,'bold'), width =7)
show_round_total_cost_actual.grid(row=5,column=5, padx=5, pady=5)

#-----------------Differ show
show_diff_total_distance_cost = Entry(left_frame3, font=('times',12,'bold'), width =7)
show_diff_total_distance_cost.grid(row=3,column=6, padx=5, pady=5)

show_diff_total_fail_cost = Entry(left_frame3,font=('times',12,'bold'), width =7)
show_diff_total_fail_cost.grid(row=4,column=6, padx=5, pady=5)


show_diff_total_cost = Entry(left_frame3,font=('times',12,'bold'), width =7)
show_diff_total_cost.grid(row=5,column=6, padx=5, pady=5)


root.mainloop()
