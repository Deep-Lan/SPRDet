# coding: utf-8


# In[ ]:


import os
import numpy as np
import cv2
import tensorflow as tf
import warnings
import copy
import tkinter as tk
import tkinter.messagebox
from tkinter.filedialog import askdirectory, asksaveasfilename
from tkinter.scrolledtext import ScrolledText
from pandas import DataFrame
from PIL import Image, ImageTk


# In[ ]:


warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# In[ ]:


def get_graph(graph_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


# In[ ]:


def run_inference_for_single_image(img, graph):
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(img, 0)})
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
    return output_dict


# In[ ]:


def get_detection(output_dict, threshold_score=0.5):
    classes_threshold = output_dict['detection_classes'][output_dict['detection_scores'] > threshold_score]
    scores_threshold = output_dict['detection_scores'][output_dict['detection_scores'] > threshold_score]
    boxes_threshold = output_dict['detection_boxes'][output_dict['detection_scores'] > threshold_score]
    classes = []
    boxes = []
    if 1 in classes_threshold:
        index_class1 = [classes_threshold == 1]
        score_max = max(scores_threshold[index_class1])
        index_maxscore = [scores_threshold == score_max]
        class_ap = classes_threshold[index_maxscore][0]
        classes.append(class_ap)
        box_ap = boxes_threshold[index_maxscore][0]
        boxes.append(box_ap)
    if 2 in classes_threshold or 3 in classes_threshold:
        index_class23 = [classes_threshold != 1]
        score_max = max(scores_threshold[index_class23])
        index_maxscore = [scores_threshold == score_max]
        class_sp = classes_threshold[index_maxscore][0]
        classes.append(class_sp)
        box_sp = boxes_threshold[index_maxscore][0]
        boxes.append(box_sp)
    classes = np.array(classes)
    boxes = np.array(boxes)
    detection = {'classes': classes, 'boxes': boxes}
    return detection


# In[ ]:


def get_ROI_radiiinterval(img_shape, detection):
    stds = [0.0023200780009173965, 0.0025253588070692063]
    height, width = img_shape[0:2]
    box_ROI, interval_ap, interval_sp = None, None, None
    if 1 in detection['classes']:
        box_ap = detection['boxes'][detection['classes'] == 1][0]
        ymin = np.maximum(height * box_ap[0] - 20, 0)
        xmin = np.maximum(width * box_ap[1] - 20, 0)
        ymax = np.minimum(height * box_ap[2] + 20, height - 1)
        xmax = np.minimum(width * box_ap[3] + 20, width - 1)
        box_ROI = (int(round(ymin)), int(round(xmin)), int(round(ymax)), int(round(xmax)))
        radius_ap = ((box_ap[2] - box_ap[0]) * height + (box_ap[3] - box_ap[1]) * width) / 2 / 2
        rmin_ap = radius_ap - 3 * stds[0] * np.minimum(height, width)
        rmax_ap = radius_ap + 3 * stds[0] * np.minimum(height, width)
        interval_ap = (int(round(rmin_ap)), int(round(rmax_ap)))
    if 2 in detection['classes'] or 3 in detection['classes']:
        box_sp = detection['boxes'][detection['classes'] != 1][0]
        radius_sp = ((box_sp[2] - box_sp[0]) * height + (box_sp[3] - box_sp[1]) * width) / 2 / 2 
        rmin_sp = radius_sp - 3 * stds[1] * np.minimum(height, width)
        rmax_sp = radius_sp + 3 * stds[1] * np.minimum(height, width)
        interval_sp = (int(round(rmin_sp)), int(round(rmax_sp)))
    return box_ROI, interval_ap, interval_sp


# In[ ]:


def get_threshold_by_MET(img):
    maxvalue = np.max(img)
    [height, width] = img.shape
    tab = np.zeros(maxvalue + 1)
    h = cv2.calcHist([img], [0], None, [256], [0, 256])
    h = h[: , 0]
    h = h / (height * width)
    p = np.sum(h[: maxvalue + 1])
    u = np.sum(h[: maxvalue + 1] * np.arange(maxvalue + 1))
    for t in range(maxvalue + 1):
        pp1, pp2, kk1, kk2 = 0, 0, 0, 0
        p1 = np.sum(h[: t + 1])
        u1 = np.sum(h[: t + 1] * np.arange(t + 1))
        if (p - p1) != 0:
            u2 = (u - u1) / (p - p1)
        else:
            u2 = 0
        if p1 != 0:
            u1 = u1 / p1
        else:
            u1 = 0
        k1 = np.sum(h[: t + 1] * (np.arange(t + 1) - u1) * (np.arange(t + 1) - u1))
        k2 = np.sum(h[t + 1:] * (np.arange(t + 1, 256) - u2) * (np.arange(t + 1, 256) - u2))
        if p1 != 0:
            k1 = np.sqrt(k1 / p1)
            pp1 = p1 * np.log(p1)
        if (p - p1) != 0:
            k2 = np.sqrt(k2 / (p - p1))
            pp2 = (p - p1) * np.log(p - p1)
        if k1 != 0:
            kk1 = p1 * np.log(k1)
        if k2 != 0:
            kk2 = (p - p1) * np.log(k2)
        tab[t] = 1 + 2 * (kk1 + kk2) - 2 * (pp1 + pp2)
    minvalue = np.min(tab)
    thresh = np.mean(np.where(tab == minvalue)) + 1
    return thresh


# In[ ]:


def get_segmentation(img):
    N=4
    thresh_global = get_threshold_by_MET(img)
    [h, w] = img.shape
    h_blk = int(round(h / N))
    w_blk = int(round(w / N))
    h_new = h_blk * N
    w_new = w_blk * N
    img_resized = cv2.resize(img, (w_new, h_new))
    thresh_block = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            img_block = img_resized[i * h_blk:(i + 1) * h_blk, j * w_blk:(j+ 1) * w_blk]
            thresh_block[i, j] = get_threshold_by_MET(img_block)
    thresh_block = np.around(thresh_block).astype(np.uint8)
    thresh_block = cv2.medianBlur(thresh_block, 3)
    thresh_local = cv2.resize(thresh_block, (w, h))
    thresh = 0.7 * thresh_local + 0.3 * thresh_global
    img_seg = img.copy()
    img_seg[img > thresh] = 255
    img_seg[img <= thresh] = 0
    return img_seg


# In[ ]:


def get_center_by_correlation(img):
    img_freq = np.fft.fft2(img)
    img_conv = np.abs(np.fft.ifft2(img_freq * img_freq))
    value_conv = np.max(img_conv)
    index_conv = np.where(img_conv == value_conv)
    y_conv = index_conv[0][0]
    x_conv = index_conv[1][0]
    if y_conv > (img.shape[0] / 2):
        y_circle = y_conv / 2
    else:
        y_circle = (y_conv + img.shape[0]) / 2
    if x_conv > (img.shape[1] / 2):
        x_circle = x_conv / 2
    else:
        x_circle = (x_conv + img.shape[1]) / 2
    center = (int(x_circle), int(y_circle))
    return center


# In[ ]:


def get_radius_by_statistics(img, center, interval, radius_ap=None):
    img_dist = np.ones(img.shape, dtype=np.uint8)
    img_dist[center[1], center[0]] = 0
    img_dist = cv2.distanceTransform(img_dist, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    img_dist = np.around(img_dist)
    mean_list = []
    if radius_ap == None:
        radius_array = np.arange(interval[0], interval[1] + 1)
        for r in radius_array:
            mean = np.mean(img[img_dist == r])
            mean_list.append(mean)
        mean_list_past = copy.deepcopy(mean_list)
        mean_list_past.pop(-1)
        mean_list_now = copy.deepcopy(mean_list)
        mean_list_now.pop(0)
        difference_array = np.array(mean_list_now) - np.array(mean_list_past)
        difference_min = np.min(difference_array)
        index_min = np.where(difference_array == difference_min)[0][0]
        radius = radius_array[index_min]
    else:
        radius_array = np.arange(interval[0], np.minimum(interval[1] + 1, radius_ap - 5))
        for r in radius_array:
            mean = np.mean(img[img_dist == r])
            mean_list.append(mean)
        mean_min = np.min(mean_list)
        index_min = np.where(mean_list == mean_min)[0][0]
        radius = radius_array[index_min]
    radius = int(radius)
    return radius


# In[ ]:


def algorithm(img, memorybox=None):
    img = img.copy()
    if memorybox == None:
        output_dict = run_inference_for_single_image(img, detection_graph)
        detection = get_detection(output_dict)
        box_ROI, interval_ap, interval_sp = get_ROI_radiiinterval(img.shape, detection)
        radius_ap, radius_sp, img_ROI = None, None, None
        if interval_ap != None:
            img_ROI = img[box_ROI[0]:box_ROI[2], box_ROI[1]:box_ROI[3]]
            img_gray = cv2.cvtColor(img_ROI, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.medianBlur(img_gray, 5)
            img_seg_ap = get_segmentation(img_gray)
            img_seg_ap = cv2.morphologyEx(img_seg_ap, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
            img_seg_ap = cv2.morphologyEx(img_seg_ap, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
            center = get_center_by_correlation(img_seg_ap)
            radius_ap = get_radius_by_statistics(img_gray, center, interval_ap)
            cv2.circle(img_ROI, center, radius_ap, (255, 0, 0), 2)
            if interval_sp != None:
                radius_sp = get_radius_by_statistics(img_gray, center, interval_sp, radius_ap)
                cv2.circle(img_ROI, center, radius_sp, (0, 0, 255), 2)
    else:
        (box_ROI, center, radius_ap, radius_sp) = memorybox
        img_ROI = img[box_ROI[0]:box_ROI[2], box_ROI[1]:box_ROI[3]]
        img_gray = cv2.cvtColor(img_ROI, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.medianBlur(img_gray, 5)
        radius_sp = get_radius_by_statistics(img_gray, center, (radius_sp - 20, radius_sp + 20), radius_ap)
        cv2.circle(img_ROI, center, radius_ap, (255, 0, 0), 2)
        cv2.circle(img_ROI, center, radius_sp, (0, 0, 255), 2)
    memorybox = (box_ROI, center, radius_ap, radius_sp)
    return radius_ap, radius_sp, img_ROI, memorybox


# In[ ]:
    

def ShowInp(flag=True):
    global img_inp
    if flag == True:
        w_frame = lf_inp.winfo_width()
        h_frame = lf_inp.winfo_height()
        w_img = img_inp.shape[1]
        h_img = img_inp.shape[0]
        f_w = w_frame / w_img
        f_h = h_frame / h_img
        f = np.minimum(f_w, f_h)
        img_show = cv2.resize(img_inp, dsize=(0, 0), fx=f, fy=f)
        img_pil = Image.fromarray(img_show)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        l_inp.imgtk = img_tk
        l_inp.config(image=img_tk)
    else:
        l_inp.imgtk = None
        l_inp.config(image=None)


# In[ ]:


def ShowDet(flag=True):
    global img_det
    if flag == True:
        w_frame = lf_det.winfo_width()
        h_frame = lf_det.winfo_height()
        w_img = img_det.shape[1]
        h_img = img_det.shape[0]
        f_w = w_frame / w_img
        f_h = h_frame / h_img
        f = np.minimum(f_w, f_h)
        img_show = cv2.resize(img_det, dsize=(0, 0), fx=f, fy=f)
        img_pil = Image.fromarray(img_show)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        l_det.imgtk = img_tk
        l_det.config(image=img_tk)
    else:
        l_det.imgtk = None
        l_det.config(image=None)


# In[ ]:


def SelectDirectory():
    try:
        var_dir.set(askdirectory())
        directory = var_dir.get()
        list_file = os.listdir(directory)
        list_img_name.clear()
        for file in list_file:
            if file.find('.') != -1:
                forma = file.split('.')[-1]
                if forma in ['bmp', 'dib', 'jpeg', 'jpg', 'jpe', 'png', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'tiff', 'tif', 'exr', 'jp2']:
                    list_img_name.append(file)
        list_img_name.sort()
        var_file_list.set(list_img_name)    
        ShowInp(flag=False)
        ShowDet(flag=False)
        var_radius_ap.set(None)
        var_radius_sp.set(None)
        var_theta.set(None)
    except:
        return


# In[ ]:


def SelectFile(event):
    global img_inp, img_det
    file = lb_file.get(lb_file.curselection())
    path_file = os.path.join(var_dir.get(), file)
    img_inp = cv2.imread(path_file)
    ShowInp()
    ShowDet(flag=False)
    var_radius_ap.set(None)
    var_radius_sp.set(None)
    var_theta.set(None)
    
    
# In[ ]:
    

def ClearText():
    list_file.clear()
    list_radius_ap.clear()
    list_radius_sp.clear()
    list_theta.clear()
    st_tex.config(state=tk.NORMAL)
    st_tex.delete(1.0, tk.END)
    st_tex.config(state=tk.DISABLED)
    
    
# In[ ]:
    
    
def InsertText(file):
    radius_ap = var_radius_ap.get()
    radius_sp = var_radius_sp.get()
    theta = var_theta.get()
    list_file.append(file)
    list_radius_ap.append(radius_ap)
    list_radius_sp.append(radius_sp)
    list_theta.append(theta)
    st_tex.config(state=tk.NORMAL)
    st_tex.insert('end', file + ' Rap:' + str(radius_ap) + ' Rsp:' + str(radius_sp) + ' theta:' + str(theta) + '\n')
    st_tex.config(state=tk.DISABLED)
    
    
# In[ ]:
    
    
def SaveText():
    path = asksaveasfilename(filetypes=[('CSV Files', '*.csv'), ('Excel Files', '*.xlsx') ,('Excel Files 97-2003', '*.xls')], defaultextension='.csv', initialfile='untitled')
    data = {'file': list_file, 'Rap': list_radius_ap, 'Rsp': list_radius_sp, 'theta': list_theta}
    df = DataFrame(data)
    print(path)
    if path == '':
        return
    elif path.find('.csv') != -1:
        df.to_csv(path)
    elif (path.find('.xlsx') != -1) or (path.find('.xls') != -1):
        df.to_excel(path)
    else:
        tk.messagebox.showerror(title='Error!!!', message='please input correct file format!')

        

# In[ ]:


def RunOnce():
    try:
        global img_inp, img_det
        radius_ap, radius_sp, img_det, _ = algorithm(img_inp)
        try:
            theta = round(np.arcsin(var_NA.get() * radius_sp / var_n.get() / radius_ap), 4)
        except:
            theta = None  
        var_radius_ap.set(radius_ap)
        var_radius_sp.set(radius_sp)
        var_theta.set(theta)
        file = lb_file.get(lb_file.curselection())
        InsertText(file)
        ShowInp()
        ShowDet()
    except:
        tk.messagebox.showerror(title='Error!!!', message='Parameter error or directory and file unselected!')


# In[ ]:
    
    
def RunAll():
    global img_inp, img_det
    files = copy.deepcopy(list_img_name)
    if len(files) == 0:
        tk.messagebox.showerror(title='Error!!!', message='Parameter error or directory unselected!')
    for file in files:
        path_file = os.path.join(var_dir.get(), file)
        img_inp = cv2.imread(path_file)
        radius_ap, radius_sp, img_det, _ = algorithm(img_inp)
        try:
            theta = round(np.arcsin(var_NA.get() * radius_sp / var_n.get() / radius_ap), 4)
        except:
            theta = None
        var_radius_ap.set(radius_ap)
        var_radius_sp.set(radius_sp)
        var_theta.set(theta)
        InsertText(file)
        ShowInp()
        ShowDet()
        win.update()


# In[ ]:


def RunAllFast():
    yesno = tk.messagebox.askyesno(title='Warning!', message='Before pushing the button "run all (fast)", please ensure all the images have clear aperture and a SPs absorption profile!\n\nAre you sure?')
    if yesno == True:
        global img_inp, img_det
        files = copy.deepcopy(list_img_name)
        if len(files) > 0:
            file = files[0]
            path_file = os.path.join(var_dir.get(), file)
            img_inp = cv2.imread(path_file)
            radius_ap, radius_sp, img_det, memorybox = algorithm(img_inp)
            try:
                theta = round(np.arcsin(var_NA.get() * radius_sp / var_n.get() / radius_ap), 4)
            except:
                theta = None
            var_radius_ap.set(radius_ap)
            var_radius_sp.set(radius_sp)
            var_theta.set(theta)
            InsertText(file)
            ShowInp()
            ShowDet()
            win.update()
            files.pop(0)
            for file in files:
                path_file = os.path.join(var_dir.get(), file)
                img_inp = cv2.imread(path_file)
                radius_ap, radius_sp, img_det, memorybox = algorithm(img_inp, memorybox)
                try:
                    theta = round(np.arcsin(var_NA.get() * radius_sp / var_n.get() / radius_ap), 4)
                except:
                    theta = None
                var_radius_ap.set(radius_ap)
                var_radius_sp.set(radius_sp)
                var_theta.set(theta)
                InsertText(file)
                ShowInp()
                ShowDet()
                win.update()
        else:
            tk.messagebox.showerror(title='Error!!!', message='Parameter error, directory unselected or chear aperture and SPR absorption profile loss!')


# In[ ]:


PATH_TO_FROZEN_GRAPH = 'model/frozen_inference_graph.pb'
detection_graph = get_graph(PATH_TO_FROZEN_GRAPH)


# In[ ]:


win = tk.Tk()
win.iconbitmap('SPRDet.ico')
win.title('SPRDet')
win.minsize(1000, 600)


var_dir = tk.StringVar()

var_file_list = tk.StringVar()

var_radius_ap = tk.IntVar()
var_radius_ap.set(None)
var_radius_sp = tk.IntVar()
var_radius_sp.set(None)
var_theta = tk.DoubleVar()
var_theta.set(None)

var_NA = tk.DoubleVar()
var_NA.set(1.25)
var_n = tk.DoubleVar()
var_n.set(1.518)

list_img_name = []

list_file = []
list_radius_ap = []
list_radius_sp = []
list_theta = []

img_inp = None
img_det = None


lf_dir = tk.LabelFrame(win, text='directory and files', labelanchor='n')
lf_dir.place(relx=0, rely=0, relwidth=0.25, relheight=0.5)
tk.Label(lf_dir, text='directory:').place(relx=0.05, rely=0, relwidth=0.25, relheight=0.1)
tk.Entry(lf_dir, textvariable=var_dir, state=tk.DISABLED).place(relx=0.3, rely=0, relwidth=0.45, relheight=0.1)
tk.Button(lf_dir, text='select', command=SelectDirectory).place(relx=0.75, rely=0, relwidth=0.2, relheight=0.1)
f_file = tk.Frame(lf_dir)
f_file.place(relx=0.05, rely=0.1, relwidth=0.9, relheight=0.9)
sb_file = tk.Scrollbar(f_file)
sb_file.pack(side=tk.RIGHT, fill=tk.Y)
lb_file = tk.Listbox(f_file, listvariable=var_file_list)
lb_file.bind('<ButtonRelease-1>', SelectFile)
lb_file.pack(side=tk.LEFT, fill=tk.BOTH, expand=1,)
sb_file.config(command=lb_file.yview)
lb_file.config(yscrollcommand=sb_file.set)


lf_rec = tk.LabelFrame(win, text='outcome records', labelanchor='n')
lf_rec.place(relx=0, rely=0.5, relwidth=0.25, relheight=0.5)
st_tex = ScrolledText(lf_rec, state=tk.DISABLED)
st_tex.place(relx=0.05, rely=0, relwidth=0.9, relheight=0.9)
tk.Button(lf_rec, text='clear', command=ClearText).place(relx=0.1, rely=0.9, relwidth=0.3, relheight=0.1)
tk.Button(lf_rec, text='save', command=SaveText).place(relx=0.6, rely=0.9, relwidth=0.3, relheight=0.1)


lf_mai = tk.LabelFrame(win)
lf_mai.place(relx=0.25, rely=0, relwidth=0.75, relheight=1)
tk.Label(lf_mai, text='V1.0').place(relx=0.90, rely=0.95, relwidth=0.1, relheight=0.05)

lf_inp = tk.LabelFrame(lf_mai, text='input image', labelanchor='n')
lf_inp.place(relx=0.05, rely=0.05, relwidth=0.45, relheight=0.55)
l_inp = tk.Label(lf_inp)
l_inp.place(relx=0, rely=0, relwidth=1, relheight=1)

lf_det = tk.LabelFrame(lf_mai, text='detection image', labelanchor='n')
lf_det.place(relx=0.55, rely=0.05, relwidth=0.4, relheight=0.55)
l_det = tk.Label(lf_det)
l_det.place(relx=0, rely=0, relwidth=1, relheight=1)

lf_par = tk.LabelFrame(lf_mai, text='parameters', labelanchor='n')
lf_par.place(relx=0.05, rely=0.65, relwidth=0.3, relheight=0.3)
tk.Label(lf_par, text='NA (numerical aperture):').place(relx=0, rely=0, relwidth=1, relheight=0.2)
tk.Entry(lf_par, textvariable=var_NA, font=('Arial', 20), justify='center').place(relx=0.2, rely=0.2, relwidth=0.6, relheight=0.25)
tk.Label(lf_par, text='n (refractive index):').place(relx=0, rely=0.5, relwidth=1, relheight=0.20)
tk.Entry(lf_par, textvariable=var_n, font=('Arial', 20), justify='center').place(relx=0.2, rely=0.70, relwidth=0.6, relheight=0.25)

lf_rad = tk.LabelFrame(lf_mai, text='outcome', labelanchor='n')
lf_rad.place(relx=0.35, rely=0.65, relwidth=0.3, relheight=0.3)
tk.Label(lf_rad, text='Rap (radius of clear aperture):').place(relx=0, rely=0, relwidth=1, relheight=0.15)
tk.Label(lf_rad, textvariable=var_radius_ap, font=('Arial', 20)).place(relx=0, rely=0.15, relwidth=1, relheight=0.15)
tk.Label(lf_rad, text='Rsp (radius of SPR absorption profile):').place(relx=0, rely=0.3, relwidth=1, relheight=0.15)
tk.Label(lf_rad, textvariable=var_radius_sp, font=('Arial', 20)).place(relx=0, rely=0.45, relwidth=1, relheight=0.15)
tk.Label(lf_rad, text='theta (SPR excitation angle) /rad:').place(relx=0, rely=0.65, relwidth=1, relheight=0.15)
tk.Label(lf_rad, textvariable=var_theta, font=('Arial', 20)).place(relx=0, rely=0.8, relwidth=1, relheight=0.15)

lf_but = tk.LabelFrame(lf_mai, text='buttons', labelanchor='n')
lf_but.place(relx=0.65, rely=0.65, relwidth=0.3, relheight=0.3)
tk.Button(lf_but, text='run once', command=RunOnce).place(relx=0.2, rely=0.01, relwidth=0.6, relheight=0.3)
tk.Button(lf_but, text='run all', command=RunAll).place(relx=0.2, rely=0.34, relwidth=0.6, relheight=0.3)
tk.Button(lf_but, text='run all (fast)', command=RunAllFast).place(relx=0.2, rely=0.67, relwidth=0.6, relheight=0.3)


win.mainloop()

