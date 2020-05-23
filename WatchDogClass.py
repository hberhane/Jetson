#!/usr/bin/env python
# coding: utf-8

# In[4]:

import subprocess
import sys
import time
import logging
#from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import LoggingEventHandler
from watchdog.events import PatternMatchingEventHandler
import re
import watchdog.events
import os
from pathlib import Path

class Patient:
    
    def __init__(self, name):
        self.num1 = 0;
        self.num2 = 0;
        self.l1 = [];
        self.l2 = [];
        self.scanName1 =""
        self.scanName2 = ""
        self.finalPath1 = ""
        self.finalPath2 = ""
        self.magPath = ""
        self.velPath = ""
        self.patientName = name
        
    def getPatientName(self):
        return self.patientName

    def getFlow(self):
        return self.velPath

    def getMag(self):
        return self.magPath
    
    #this assumes that the scan_name has passed our regex filter and we just need to identify variables to increment    
    def takeInScan(self,scan_name, full_Path):
        #print("inputed name = " + scan_name)
        #print("first saved Name = " + self.scanName1)
        #print("second saved Name = " + self.scanName2)
        if(scan_name == self.scanName1):    #it matches our first scan
            self.l1.append('1');
            self.num1 = len(self.l1)
            #print(len(self.l1))
        elif(scan_name == self.scanName2):  #it matches our second
            self.l2.append('1');
            self.num2 = len(self.l2)
            #print(len(self.l2))
        elif(self.scanName1 == ""):          #we have no scans yet
            self.scanName1 = scan_name
            self.l1.append('1')
            self.num1 = len(self.l1)
            self.finalPath1 = full_Path.pop()
        elif(self.scanName2 == ""):          #we have 1 scan, it didn't match, this must be our second
            self.scanName2 = scan_name
            self.l2.append('1');
            self.num2 = len(self.l2)
            self.finalPath2 = full_Path.pop()
        else:                               #we have 2 scans all ready, we could either keep track of all the scans that match (Refactor this class)                 
            #print('conflict')               #or assume the previous scans have all ready finshed.  If they don't meet push criteria, one of them is a 
                                            #probably the scout scan, get rid of it [this assumes 1 patient only gets on 4D flow scan]
            if(self.num1 > 0 and self.num2 > 0):  #this must always be true as it passed above conditions
                if(not (self.num1/self.num2 == 3 or self.num2/self.num1 == 3)):  #we do not meet push criteria
                    if(self.num1 < self.num2):
                        self.num1 = 1
                        self.scanName1 = scan_name
                        self.finalPath1 = full_Path.pop()
                        self.l1 = []
                        self.l1.append('1')
                    else:
                        self.num2 = 1
                        self.scanName2 = scan_name
                        self.finalPath2 = full_Path.pop()
                        self.l2 = []
                        self.l2.append('1')
            #might need to swap 2 and 1 if 2 > 1 and add to 2
        #print("First num is : " + str(self.num1))
        #print("Second num is : " + str(self.num2))

                
                    
            
    
    def pushToJetson(self):
        if(len(self.l1) !=0 and len(self.l2) != 0 and len(self.l1) + len(self.l2) > 1000):
            if(len(self.l1)/len(self.l2) == 3):
                print("made it to case 1")
                self.magPath = self.finalPath2
                self.velPath = self.finalPath1
                print(len(self.l1))

                #strip of dicomfile and get filepath to folder
            
                return True
            elif(len(self.l2)/len(self.l1) == 3):
                print("made it to case 2")
                self.magPath = self.finalPath1
                self.velPath = self.finalPath2
                print(len(self.l2))
                
                return True
            else:
                return False


class ScanFolder:
    'Class defining a scan folder'
    
    def __init__(self):
        self.FourDFlowScanNameC1 = ""
        self.FourDFlowScanNameC2 = ""
        self.DicomNamesC1 = []
        self.DicomNamesC2 = []
        self.Candidate_Num1 = 0
        self.Candidate_Num2 = 0
        self.finalPath1 = ""
        self.finalPath2 = ""
        self.Patients = []

        self.path = r'/mnt/data_imaging/jetsonTest'
        #self.path = r'/mnt/data_imaging/dicom'
        self.documents = dict() # key = document label   value = Document reference

        self.patterns = "*"
        self.ignore_patterns = ""
        self.ignore_directories = False
        self.case_sensitive = True
        self.my_event_handler = watchdog.events.PatternMatchingEventHandler(self.patterns, self.ignore_patterns, self.ignore_directories, self.case_sensitive)
        #self.my_event_handler.on_any_event = self.on_any_event
        #self.my_event_handler.on_deleted = self.on_deleted
        self.my_event_handler.on_created = self.on_created
        
        self.observer = Observer()
        self.observer.schedule(self.my_event_handler, self.path, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.observer.stop()
            self.observer.join()

   # def on_any_event(self, event):
   #     print(event.src_path, event.event_type)
   #     print("self even")

    #def stop(self):
    #    self.observer.stop()
    #    self.observer.join()
        
    def on_deleted(self,event):
        print(f"what the f**k! Someone deleted {event.src_path}!")
        #print(self.FourDFlowScanNameC1)
        #self.FourDFlowScanNameC1 = "dogs"
    
    def on_created(self,event):
        #print(f"hey, {event.src_path} has been created!")
        fullFilePath = {event.src_path}
        parts = str({event.src_path}.pop()).split('/')   #change to be / for jetson and \ for my desktop
        #print(parts)
        #print(len(parts))
        ##need to try catch this in case of weird inputs?
        fileName = parts[-1]
        scanName = parts[-2]
        patientName = parts[-3]
    
        #if "fileName" doesn't have a '.extension' then it is a folder OR use os.path.isfile() / os.path.isdir()
        fileNameParts = fileName.split('.')
        #print(fileNameParts)
        numParts = len(fileNameParts)
        #print(numParts)
        #print("The scan name")
        #print(scanName)
        
        
        if(numParts >=1):
            #pattern matching block, swap from 4D to 4D and Aorta
            pattern = "4D"
            #print("our matching result :" + str( re.search(pattern, scanName)))
            result = re.search(pattern, scanName)
            
            pattern2 = "[aA][oO][rR][tT][aA]"
            result2 = re.search(pattern2, scanName)
            
            if(result and result2):
                patientFound = False
                for patient in self.Patients:
                    pName = patient.getPatientName()
                    if(pName == patientName):
                        patientFound = True
                        patient.takeInScan(scanName, fullFilePath)
                        if(patient.pushToJetson()):
                            print("Let's push to Jetson!")
                            flow = patient.getFlow()
                            flow = Path(flow).parent
                            print(flow)
                            mag = patient.getMag()
                            mag = Path(mag).parent

                            cmd = ["python","./mrStructWrite.py","--flow="+str(flow),"--mag="+str(mag)]
                            #subprocess.call(["python /home/condauser/Documents/ao_seg/4dflow_dicom_to_array.py --flow=",str(flow)," --mag=",str(mag)])
                            subprocess.call(cmd)
                            #cmdstr = re.sub(r"(",r"\(","python /home/condauser/Documents/ao_seg/4dflow_dicom_to_array.py --flow="+str(flow)+" --mag="+str(mag))
                            #print(cmdstr)
                            #os.system(cmdstr)
                            #delete the patient
                            
                if(not patientFound):
                    #print("Adding patient " + str(patientName))
                    self.Patients.append( Patient(patientName) )
                    self.Patients[-1].takeInScan(scanName, fullFilePath)

        
        

sf = ScanFolder()













