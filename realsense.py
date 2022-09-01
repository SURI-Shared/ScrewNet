'''
Created on Sep 1, 2022

@author: ggutow
'''
import pyrealsense2 as rs
import numpy as np
import time
import torch

from models import ScrewNet

def record_frames(dur):
    pipeline=rs.pipeline()
    align=rs.align(rs.stream.color)
    #device=rs.config().resolve(rs.pipeline_wrapper(pipeline)).get_device()
    pipeline.start()
    start=time.perf_counter()
    dimages=[]
    cimages=[]
    try:
        while time.perf_counter()-start<dur:
            frames=pipeline.wait_for_frames()
            aligned_frames=align.process(frames)
            depthf=aligned_frames.get_depth_frame()
            colorf=aligned_frames.get_color_frame()
            if not depthf or not colorf:
                continue
            dimages.append(np.array(depthf.get_data()))
            cimages.append(np.array(colorf.get_data()))
    finally:
        pipeline.stop()
        print(len(dimages))
    return cimages,dimages

def process_realsense(duration):
    cimages,dimages=record_frames(duration)
    depth=torch.Tensor(np.tile(np.expand_dims(np.array(dimages).astype(np.int16),(0,2)),(1,1,3,1,1)))    
    screwnet=ScrewNet(lstm_hidden_dim=1000,n_lstm_hidden_layers=1,n_output=8)
    screwnet.load_state_dict(torch.load("data/trained_wts/nnv1_partnet_combined_screw.net"))
    device=torch.device(0)
    screwnet.float().to(device)
    screwnet.eval()
    with torch.no_grad():
        depthcuda=depth.to(device)
        prediction=screwnet(depthcuda)
        prediction=prediction.view(prediction.size(0),-1,8)
        prediction=prediction[0,1:,:]
        prediction=prediction.cpu()
    return cimages,dimages,depth,prediction
