#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


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
        
        
        self.scanNames = [];
        self.numbers = [];
        self.paths = [];
        self.lengths = [];
        
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

    
   
                    
    #new idea just get the directory and find the length of all the directories we have every creation event
    def takeInScan3(self,scan_name,full_Path):
        #print(full_Path)
        if(len(self.scanNames) == 0):
            print('adding it initally')
            self.scanNames.append(scan_name)
            self.paths.append(full_Path.pop())
        else:
            nameFound = False
            for i in range(0, len(self.scanNames)):
                names = self.scanNames
                name = names[i]             
                if(scan_name == name):
                    nameFound = True
            if(not nameFound):
                self.scanNames.append(scan_name)
                self.paths.append(full_Path.pop())
            
            
    
    
    
    def pushToJetson3(self):
        #print("Jetson Time?!")
        print(self.scanNames)
        #print(len(self.scanNames))
        for i in range(0, len(self.scanNames)):
            for j in range(0, len(self.scanNames)):
                if(i != j):
                    path1 = self.paths[i]
                    path2 = self.paths[j]
                    #strip off last filename
                    path1 = path1.split("/")
                    path1 = path1[:-1]
                    p1 = ""
                    for e in path1:
                        p1 = p1 + e + "/"
                        
                    path2 = path2.split("/")
                    path2 = path2[:-1]
                    p2 = ""
                    for e in path2:
                        p2 = p2 + e + "/"
                    #print("The paths")
                    p1 = p1 + "/"
                    p2 = p2 + "/"
                    #print(p1)
                    #print(p2)
                    list = os.listdir(p1) # dir is your directory path
                    num1 = len(list)
                    list = os.listdir(p2) # dir is your directory path                    
                    num2 = len(list)

                    #print(num2)

                    if(num1 != 0 and num2 != 0 and (num1 > 1000 or num2 > 1000)):
                        if(num1 / num2 == 3):
                            self.magPath = self.paths[j]
                            self.velPath = self.paths[i]
                            return True
                        elif(num2/num1 == 3):
                            self.magPath = self.paths[j]
                            self.velPath = self.paths[i]
                            return True                 
        return False


# In[3]:



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
        self.ProcessedPatients = []

        self.path = r'/mnt/data_imaging/jetsonTest'
        #self.path = r'C:\Users\jjb1487\Desktop\WatchdogTest'
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
        print(f"hey, {event.src_path} has been created!")
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
                #print("matched")
                for patient in self.Patients:
                    noProcess = False
                    pName = patient.getPatientName()
                    #make sure we haven't sent all ready
                    for processedName in self.ProcessedPatients:
                        if pName == processedName:
                            noProcess = True
                    if(pName == patientName and not noProcess):
                        patientFound = True
                        #print(f"hey, {event.src_path} has been created!")
                        patient.takeInScan3(scanName, fullFilePath)
                        if(patient.pushToJetson3()):
                            print("Let's push to Jetson!")
                            flow = patient.getFlow()
                            flow = Path(flow).parent
                            print(flow)
                            mag = patient.getMag()
                            mag = Path(mag).parent
                            print(mag)
                            self.ProcessedPatients.append(patient.getPatientName())
                            cmd = ["python","./mrStructWrite.py","--flow="+str(flow),"--mag="+str(mag)]
                            #subprocess.call(["python /home/condauser/Documents/ao_seg/4dflow_dicom_to_array.py --flow=",str(flow)," --mag=",str(mag)])
                            subprocess.call(cmd)
                            #cmdstr = re.sub(r"(",r"\(","python /home/condauser/Documents/ao_seg/4dflow_dicom_to_array.py --flow="+str(flow)+" --mag="+str(mag))
                            #print(cmdstr)
                            #os.system(cmdstr)
                            #delete the patient
                            #and then add them to a processed list so you don't keep processing
                            
                if(not patientFound):
                    #print("Adding patient " + str(patientName))
                    self.Patients.append( Patient(patientName) )
                    self.Patients[-1].takeInScan(scanName, fullFilePath)
                    
#        for i in range(0, len(self.processedPatients)):
#            for j in range(0, len(self.Patients)):
#                if(self.processPatients[i] == self.Patients[j].getPatientName()):
#                    print("Popping patient")
#                    self.Patients.pop[j]
        


# In[ ]:


sf = ScanFolder()


# In[ ]:




