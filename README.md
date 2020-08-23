1. pip install python2/3

2. pip install pytorch

3. see dict.txt and build model_outdoor/input/output files 
    
   you can revise the input of dict.txt:
   (1) datasetname:list
       input_path:/home/yunpengwu/dehaze/data/3.jpg /home/yunpengwu/dehaze/data/5.jpg
   
   (2) datasetname:folder
       input_path:./data ./data1

4. download "netG_epoch_2.pth" and put it into model_outdoor 
(model please connect to anthor email 16114225@bjtu.edu.en)

5. python dehaze.py 

