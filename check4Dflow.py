import time
import os
from glob import glob
import re
import subprocess
from pathlib import Path
from multiprocessing import Process

def listdirs(folder):
    return [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

# Set the folder to watch
#rootFolder = os.path.join("C:", os.sep, "Users", "Mike", "Desktop", "checkTest")
rootFolder = r"/mnt/data_imaging/jetsonTest"
sleepTime = 10.0

starttime=time.time()
while True:
  #Get a list of all folders in the dicom folder
  subFolders = glob(os.path.join(rootFolder, "*", ""))
  
  for i in subFolders:
    #print(i)
    #ptScans = glob(os.path.join(i, "*", ""))
    ptScans = listdirs(i)
    #Check for 3dpcML
    if any("3dpc_ml" in s for s in ptScans): 
        continue
        
    #Check for aorta 4D flow
    aortaScans = []
    dicomNumber = []
    for j in ptScans:
        if re.search("4D",j) and re.search("[aA][oO][rR][tT][aA]",j):
            # Get the names of the aorta 4D folders
            aortaScans.append(j)
            # Get the number of dicoms in the folder
            dcms = glob(os.path.join(i, j, "*.dcm"))
            dicomNumber.append(len(dcms))
    
    if len(aortaScans) == 0:
        continue
    
    # Check if any have 1/3 the maximum number
    velNum = max(dicomNumber)#,key=lambda item:item[1])
    velIdx = dicomNumber.index(velNum)
    if velNum > 999 and int(velNum/3) in dicomNumber:
        magIdx = dicomNumber.index(int(velNum/3))
    else:
        continue
    
    #print("Index", index)
    #print("vel", aortaScans[velIdx])
    flow = os.path.join(i,aortaScans[velIdx])
    #print("mag", aortaScans[magIdx])
    mag = os.path.join(i,aortaScans[magIdx])
    
    # Send to Jetson
    print("Send to Jetson: ",i)
    cmd = ["python","./mrStructWrite.py","--flow="+str(flow),"--mag="+str(mag)]
    cmd2 = ["python","alias_correc.py"]
    try:   
        # Make the output folder (will cause the pt to skip if failed)
        os.mkdir(os.path.join(i,"3dpc_ml"))
        p = Process(target=subprocess.call(cmd))
        p.daemon = True
        p.start()
        #subprocess.call(cmd2)
        j = Path(flow).parent
        copyfile("/home/condauser/Documents/ao_seg/MK/vel_struct.mat",os.path.join(os.path.sep,j,"3dpc_ml","vel_struct.mat"))
        
    except:
        print("Jetson crashed... :(")
        #os.mkdir(os.path.join(i,"3dpc_ml"))
    
    break
  print("tick")
  time.sleep(sleepTime - ((time.time() - starttime) % sleepTime))

