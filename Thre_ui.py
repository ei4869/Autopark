#hsv阈值获取和普通二值化阈值获取
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from tkinter import filedialog

def select_image():
    global panelA, panelB, path
    path = filedialog.askopenfilename()

    if len(path) > 0:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        if panelB is None:
            panelB = tk.Label(image=image)
            panelB.image = image
            panelB.pack(side="left", padx=10, pady=10)
        else:
            panelB.configure(image=image)
            panelB.image = image

def threshold_image(image, lower, upper):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    result = mask
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return result

def threshold_image_gray(image, threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, result = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return result

def update_image(*args):
    global panelA, path
    image = cv2.imread(path)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if v.get() == 1:
        lower = (lower_h.get(), lower_s.get(), lower_v.get())
        upper = (upper_h.get(), upper_s.get(), upper_v.get())
        result = threshold_image(image_bgr, lower, upper)
    else:
        threshold = threshold_value.get()
        result = threshold_image_gray(image_bgr, threshold)

    result = Image.fromarray(result)
    result = ImageTk.PhotoImage(result)

    if panelA is None:
        panelA = tk.Label(image=result)
        panelA.image = result
        panelA.pack(side="left", padx=10, pady=10)
    else:
        panelA.configure(image=result)
        panelA.image = result

def save_thresholds():
    lower = [lower_h.get(), lower_s.get(), lower_v.get()]
    upper = [upper_h.get(), upper_s.get(), upper_v.get()]
    threshold_string = f"[{lower},{upper}]"
    root.clipboard_clear()
    root.clipboard_append(threshold_string)
    messagebox.showinfo("Thresholds", f"Copied to clipboard: {threshold_string}")

root = tk.Tk()
root.geometry('1024x1024')
panelA = None
panelB = None

frame = tk.Frame(root)
frame.pack(side="top", fill="x")

lower_h = tk.IntVar(root)
lower_s = tk.IntVar(root)
lower_v = tk.IntVar(root)

upper_h = tk.IntVar(root)
upper_s = tk.IntVar(root)
upper_v = tk.IntVar(root)

threshold_value = tk.IntVar(root)

v = tk.IntVar(root)
v.set(1)

lower_h.set(0)
lower_s.set(0)
lower_v.set(0)

upper_h.set(255)
upper_s.set(255)
upper_v.set(255)

threshold_value.set(127)

lower_h.trace("w", update_image)
lower_s.trace("w", update_image)
lower_v.trace("w", update_image)

upper_h.trace("w", update_image)
upper_s.trace("w", update_image)
upper_v.trace("w", update_image)

threshold_value.trace("w", update_image)

btn = tk.Button(frame, text="Select an image", command=select_image)
btn.pack(side="left", padx=10, pady=10)

save_btn = tk.Button(frame, text="Save thresholds", command=save_thresholds)
save_btn.pack(side="left", padx=10, pady=10)

tk.Radiobutton(frame, text="HSV Threshold", variable=v, value=1).pack(side="left")
tk.Radiobutton(frame, text="Gray Threshold", variable=v, value=2).pack(side="left")

frame1 = tk.Frame(root)
frame1.pack(side="left")
frame2 = tk.Frame(root)
frame2.pack(side="left")
frame3 = tk.Frame(root)
frame3.pack(side="left")
frame4 = tk.Frame(root)
frame4.pack(side="left")

lower_h_label = tk.Label(frame1, text="Lower H")
lower_h_scale = tk.Scale(frame1, from_=0, to=255, variable=lower_h, orient="horizontal")

upper_h_label = tk.Label(frame1, text="Upper H")
upper_h_scale = tk.Scale(frame1, from_=0, to=255, variable=upper_h, orient="horizontal")

lower_s_label = tk.Label(frame2, text="Lower S")
lower_s_scale = tk.Scale(frame2, from_=0, to=255, variable=lower_s, orient="horizontal")

upper_s_label = tk.Label(frame2, text="Upper S")
upper_s_scale = tk.Scale(frame2, from_=0, to=255, variable=upper_s, orient="horizontal")

lower_v_label = tk.Label(frame3, text="Lower V")
lower_v_scale = tk.Scale(frame3, from_=0, to=255, variable=lower_v, orient="horizontal")

upper_v_label = tk.Label(frame3, text="Upper V")
upper_v_scale = tk.Scale(frame3, from_=0, to=255, variable=upper_v, orient="horizontal")

threshold_label = tk.Label(frame4, text="Gray Threshold")
threshold_scale = tk.Scale(frame4, from_=0, to=255, variable=threshold_value, orient="horizontal")

lower_h_label.pack()
lower_h_scale.pack()
upper_h_label.pack()
upper_h_scale.pack()

lower_s_label.pack()
lower_s_scale.pack()
upper_s_label.pack()
upper_s_scale.pack()

lower_v_label.pack()
lower_v_scale.pack()
upper_v_label.pack()
upper_v_scale.pack()

threshold_label.pack()
threshold_scale.pack()

root.mainloop()
