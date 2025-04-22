import sqlite3
import hashlib
import datetime
import MySQLdb
from flask import session
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
import numpy as np
import ast 
import os
 
import cv2
import pandas as pd

from datetime import datetime
 
 

def db_connect(): 
    _conn = MySQLdb.connect(host="localhost", user="root",
                            passwd="root", db="wpdb")
    c = _conn.cursor()

    return c, _conn

# -------------------------------register-----------------------------------------------------------------
def Buyer_reg(username,email,password):
    try:
        status=check(username)
        if status==1:
            return 0
        c, conn = db_connect()
        print(username, password, email)
        j = c.execute("INSERT INTO user(username, email, password) VALUES (%s, %s, %s)", 
               (username, email, password))
        conn.commit()
        conn.close()
        print(j)
        return j
    except Exception as e:
        print(e)
        return(str(e))
    
     
# -------------------------------------Login --------------------------------------
def Buyer_loginact(username, password):
    try:
        c, conn = db_connect()
        j = c.execute("select * from user where username='" +
                      username+"' and password='"+password+"'")
        data = c.fetchall()
        print(data)     
       
        c.fetchall()
        conn.close()
        return j
    except Exception as e:
        return(str(e))

# -------------------------------register-----------------------------------------------------------------
def user_reg(username,email,password):
    try:
        status=check1(username)
        if status==1:
            return 0
        c, conn = db_connect()
        print(username, password, email)
        j = c.execute("INSERT INTO user1(username, email, password) VALUES (%s, %s, %s)", 
               (username, email, password))
        conn.commit()
        conn.close()
        print(j)
        return j
    except Exception as e:
        print(e)
        return(str(e))
    
     
# -------------------------------------Login --------------------------------------
def user_loginact(username, password):
    try:
        c, conn = db_connect()
        j = c.execute("select * from user1 where username='" +
                      username+"' and password='"+password+"'")
        data = c.fetchall()
        print(data)     
       
        c.fetchall()
        conn.close()
        return j
    except Exception as e:
        return(str(e))

def check(username):
    try:
        c, conn = db_connect()
        j = c.execute("select * from user where username='" +
                      username+"'")
        data = c.fetchall()
        print(data)     
       
        c.fetchall()
        conn.close()
        return j
    except Exception as e:
        return(str(e))
def check1(username):
    try:
        c, conn = db_connect()
        j = c.execute("select * from user1 where username='" +
                      username+"'")
        data = c.fetchall()
        print(data)     
       
        c.fetchall()
        conn.close()
        return j
    except Exception as e:
        return(str(e))

def poniti(username, points,p):
    try:
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Convert datetime properly
        
        c, conn = db_connect()
        
        # Ensure `points` is a string without tab characters
        if isinstance(points, list):  
            points = ",".join(map(str, points))  # Convert list to comma-separated string
        elif isinstance(points, str):
            points = points.replace("\t", ",")  # Replace tab characters with commas
        
        print(username, points, current_time)
        
        # Insert query with correct values
        j = c.execute(
            "INSERT INTO points(username, points,p) VALUES (%s, %s,%s)", 
            (username, points,p)
        )
        
        conn.commit()
        conn.close()
        print(j)
        return j
    except Exception as e:
        print(e)
        return str(e)

def ad(username,lin,s):
    try:
        c, conn = db_connect()
        print(username,lin)
        j = c.execute("INSERT INTO vs(username, link,score) VALUES (%s, %s,%s)", 
               (username,lin,s))
        conn.commit()
        conn.close()
        print(j)
        return j
    except Exception as e:
        print(e)
        return(str(e))
    


def get_user_data(username):
    try:
        c, conn = db_connect()

        # Fetch all rows for 'points' table
        c.execute("SELECT * FROM points WHERE username=%s ORDER BY id DESC", (username,))
        points_data = c.fetchall()

        # Fetch all rows for 'vs' table
        c.execute("SELECT * FROM vs WHERE username=%s ORDER BY id DESC", (username,))
        vs_data = c.fetchall()

        conn.close()

        if not points_data and not vs_data:
            return None

        points_list = [{
            "id": row[0],
            "username": row[1],
            "points": row[2],  # Points stored as a string
            "current_time": row[3],
            "p": row[4]  # Performance metric
        } for row in points_data]

        vs_list = [{
            "id": row[0],
            "username": row[1],
            "link": row[2],
            "current_time": row[3],
            "score": ast.literal_eval(row[4])  # Convert string to dictionary safely
        } for row in vs_data]

        # Extract scores from 'vs' table
        blinks_values = [vs["score"].get("blinks", 0) for vs in vs_list]
        yawns_values = [vs["score"].get("yawns", 0) for vs in vs_list]
        face_angle_values = [vs["score"].get("face_angle_changes", 0) for vs in vs_list]

        # Calculate averages (handle empty lists)
        blinks_avg = np.mean(blinks_values) if blinks_values else 0
        yawns_avg = np.mean(yawns_values) if yawns_values else 0
        face_angle_avg = np.mean(face_angle_values) if face_angle_values else 0
        es="Engaged"

        # Determine engagement status
        exceeded_statuses = []
        if blinks_avg > 5:
            exceeded_statuses.append("Frustrated")
            es="Not Engaged"
        if yawns_avg > 200:
            exceeded_statuses.append("Bored")
            es="Not Engaged"
        if face_angle_avg > 5:
            exceeded_statuses.append("Confused")
            es="Not Engaged"

        engagement_status = "Engaged" if not exceeded_statuses else exceeded_statuses[0]

        return {
            "points": points_list,
            "vs": vs_list,
            "averages": {
                "blinks_avg": round(blinks_avg, 2),
                "yawns_avg": round(yawns_avg, 2),
                "face_angle_avg": round(face_angle_avg, 2),
                "engagement_status": engagement_status
            }
        },es

    except Exception as e:
        return str(e)




if __name__ == "__main__":
    print(db_connect())