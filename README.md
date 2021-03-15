# SPRDet

## Background

This is a surface plasmon resonance(SPR) excitation angle detection software based on deep learning and pattern recognition. It was developed for users engaged in nano-optical detection technology, especially in the features extraction of 2D SPR excitation angle. The commonly used SPR coupling types mainly include prismatic mode and high numerical aperture microscope mode. For the high numerical aperture microscope type, the features of the best excitation angle are represented by a set of symmetrical absorption arcs or circles formed on the back focal plane(BFP) behind the len, so it is important to precisely locate the position of the absorption arcs or circles to extract the features. Fig.1 shows the surface plasmon microscope system.

<div  align="center">    
<img src="https://github.com/Deep-Lan/SPRDet/blob/main/figures/fig1.png" />

Fig.1 Surface plasmon microscope system
</div>

## Algorithm

The software provides a complete solution for the detection of excitation angles represented by the 2D SPR absorption spectral. The solution process is as follows. Input the BFP images into the trained Faster R-CNN neural network to obtain the classification and rough position information of the aperture and SPR absorption spectrum. Then use the minimum error method to segment the BFP images. And use the two-dimensional self-convolution method to get accurate center positions of the aperture and the absorption spectrum. And then combined with the rough radius information provided by the neural network, product grayscale statistics alone radius on the BFP image to determine the accurate aperture radius and SPR absorption radius. Finally, according to the Abbe's sine formula, we can calculate the SPR excitation angle with parameters such as the numerical aperture NA, refractive index n and radius of aperture and the absorption spectrum. Fig.2 shows the algorithm procedure.

<div  align="center">    
<img src="https://github.com/Deep-Lan/SPRDet/blob/main/figures/fig2.png" />
  
Fig.2 Detection algorithm
</div>

## Software

Fig.3 shows the software GUI. It is the main interface of 2D SPR excitation angle position recognition software, which is divided into directory and file module (Fig.3-A area), result recording module (Fig.3-B area), input image display module (Fig.3-C area), detection image display module (Fig.3-D area), parameter setting module (Fig.3-E area), display module (Fig.3-F area) and button module (Fig.3-G area).

<div  align="center">    
<img src="https://github.com/Deep-Lan/SPRDet/blob/main/figures/fig3.jpg" />
  
Fig.2 Detection algorithm
</div>

