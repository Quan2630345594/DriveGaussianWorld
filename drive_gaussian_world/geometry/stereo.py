"""CPU stereo reference; observed surfaces, never obstacle/occupancy ground truth."""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from drive_gaussian_world.data.schema import TOKEN_DIM

GRID_H,GRID_W=4,16

def grid_pixels(height,width):
    ys=((np.arange(GRID_H)+0.5)*height/GRID_H).astype(int)
    xs=((np.arange(GRID_W)+0.5)*width/GRID_W).astype(int)
    return np.stack(np.meshgrid(xs,ys),-1).reshape(-1,2)

def stereo_tokens(left,right,K,baseline,width=512):
    left=np.asarray(Image.open(left).convert('RGB'))
    right=np.asarray(Image.open(right).convert('RGB'))
    h0,w0=left.shape[:2]
    h=round(h0*width/w0)
    l=cv2.resize(left,(width,h));r=cv2.resize(right,(width,h))
    sx,sy=width/w0,h/h0
    k=K.copy();k[0]*=sx;k[1]*=sy
    matcher=cv2.StereoSGBM_create(minDisparity=0,numDisparities=64,blockSize=5,
        P1=8*25,P2=32*25,disp12MaxDiff=1,uniquenessRatio=10,speckleWindowSize=50,
        speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disparity=matcher.compute(cv2.cvtColor(l,cv2.COLOR_RGB2GRAY),cv2.cvtColor(r,cv2.COLOR_RGB2GRAY)).astype(np.float32)/16
    # Left-right consistency excludes many occluded stereo surfaces.
    reverse=cv2.StereoSGBM_create(minDisparity=-64,numDisparities=64,blockSize=5,
        P1=8*25,P2=32*25,uniquenessRatio=10,speckleWindowSize=50,speckleRange=2)
    dr=reverse.compute(cv2.cvtColor(r,cv2.COLOR_RGB2GRAY),cv2.cvtColor(l,cv2.COLOR_RGB2GRAY)).astype(np.float32)/16
    yy,xx=np.indices(disparity.shape)
    xr=np.clip(np.rint(xx-disparity).astype(int),0,width-1)
    consistent=(disparity>1)&(xx-disparity>=0)&(np.abs(disparity+dr[yy,xr])<1.5)
    depth=np.divide(k[0,0]*baseline,disparity,out=np.zeros_like(disparity),where=consistent)
    depth[(depth<1)|(depth>80)]=0
    tokens=np.zeros((64,TOKEN_DIM),np.float32);mask=np.zeros(64,bool)
    for i,(u,v) in enumerate(grid_pixels(h,width)):
        y0,y1=max(0,v-5),min(h,v+6);x0,x1=max(0,u-5),min(width,u+6)
        patch=depth[y0:y1,x0:x1];valid=patch[patch>0]
        if len(valid)<12:continue
        z=float(np.median(valid));x=(u-k[0,2])*z/k[0,0];y=(v-k[1,2])*z/k[1,1]
        scale=max(0.02,z/k[0,0]*2)
        confidence=len(valid)/patch.size
        tokens[i]=[* (np.array([x,y,z])/40),*(l[v,u]/255),0.8,scale/5,scale/5,scale/5,
                   1,0,0,0,0,confidence]
        mask[i]=True
    return tokens,mask,depth
