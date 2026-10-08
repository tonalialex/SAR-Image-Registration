"""Restrict genuine SAR-Harris extrema to independently detected target support.

The supplied SAR-SIFT core is unchanged. This detector uses its response,
orientation and descriptor functions with the same strict local-max rule.
"""
import cv2,numpy as np
from sar_registration import sar_sift_core as core
class MaskedSARSIFT(core.SARSIFT):
    def __init__(self,config,support):
        super().__init__(config);self.support=np.asarray(support,bool)
    def detect_and_compute(self,image):
        image=core._gray_float(image);height,width=image.shape
        if self.support.shape!=image.shape:raise ValueError("Mask grid mismatch")
        gradients=[];angles=[];candidates=[]
        for layer in range(self.config.layers):
            scale=self.config.sigma*self.config.ratio**layer
            response,gradient,angle=core._sar_scale(image,scale,self.config.harris_d)
            gradients.append(gradient);angles.append(angle)
            center=response[2:-2,2:-2]
            keep=self.support[2:-2,2:-2]&(center>self.config.harris_threshold)
            for dy in [-1,0,1]:
                for dx in [-1,0,1]:
                    if dx or dy:keep&=center>response[2+dy:height-2+dy,2+dx:width-2+dx]
            ys,xs=np.where(keep)
            for y,x in zip(ys+2,xs+2):
                histogram=core._orientation_hist(gradient,angle,x+1,y+1,scale,self.config.orientation_bins)
                threshold=.8*float(histogram.max(initial=0.))
                for bin_index in range(len(histogram)):
                    left=histogram[(bin_index-1)%len(histogram)];right=histogram[(bin_index+1)%len(histogram)]
                    if not(histogram[bin_index]>left and histogram[bin_index]>right and histogram[bin_index]>threshold):continue
                    denominator=left+right-2*histogram[bin_index]
                    offset=.5*(left-right)/denominator if abs(denominator)>1e-12 else 0.
                    peak=bin_index+offset
                    if peak<0:peak+=len(histogram)
                    elif peak>=len(histogram):peak-=len(histogram)
                    candidates.append((float(response[y,x]),int(x+1),int(y+1),scale,(360./len(histogram))*peak,layer+1))
        candidates.sort(key=lambda x:x[0],reverse=True)
        if self.config.max_features is not None:candidates=candidates[:self.config.max_features]
        points=[];descriptors=[]
        for _,x,y,scale,main_angle,layer in candidates:
            points.append((float(x-1),float(y-1)))
            descriptors.append(core._descriptor(gradients[layer-1],angles[layer-1],x,y,scale,main_angle,
                                               self.config.descriptor_spatial_bins,self.config.descriptor_angular_bins))
        return np.asarray(points,np.float32).reshape(-1,2),np.asarray(descriptors,np.float32).reshape(-1,136)
