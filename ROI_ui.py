#获取ROI参数
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox

class ImageCanvas(tk.Canvas):
    def __init__(self, parent, **kwargs):
        tk.Canvas.__init__(self, parent, **kwargs)
        self.bind("<Configure>", self.redraw)
        self.bind('<ButtonPress-1>', self.drag_start)
        self.bind('<B1-Motion>', self.drag_move)
        self.bind('<MouseWheel>', self.zoom)  # Windows and MacOS
        self.bind('<Button-4>', self.zoom)  # Linux scroll up
        self.bind('<Button-5>', self.zoom)  # Linux scroll down
        self.image = None
        self.scale = 1.0

    def drag_start(self, event):
        self.scan_mark(event.x, event.y)

    def drag_move(self, event):
        self.scan_dragto(event.x, event.y, gain=1)

    def zoom(self, event):
        if self.image:
            scale_increment = 0.1 * (1 if event.delta > 0 else -1)
            self.scale += scale_increment
            self.scale = max(0.1, self.scale)
            self.redraw()

    def open_image(self, path):
        self.image = Image.open(path)
        self.orig_width, self.orig_height = self.image.size
        self.width, self.height = self.orig_width, self.orig_height
        self.config(scrollregion=(0,0, self.width, self.height))
        self.display_image()

    def display_image(self):
        resized = self.image.resize((self.width, self.height))
        self.photo = ImageTk.PhotoImage(resized)
        self.create_image(0,0, image=self.photo, anchor='nw')

    def redraw(self, event=None):
        if self.image:
            self.width, self.height = int(self.orig_width * self.scale), int(self.orig_height * self.scale)
            self.config(scrollregion=(0, 0, self.width, self.height))
            self.display_image()

class ROIEditor:
    def __init__(self, master):
        self.master = master
        self.canvas = ImageCanvas(master)
        self.canvas.pack(fill='both', expand=True)
        self.rect_id = None
        self.roi_mode = False

        # Create menu
        menubar = tk.Menu(master)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open", command=self.open_image)
        filemenu.add_command(label="Save ROI", command=self.save_roi)
        menubar.add_cascade(label="File", menu=filemenu)
        master.config(menu=menubar)

        # Create button
        self.roi_button = tk.Button(master, text="ROI Selection Mode", command=self.toggle_roi_mode)
        self.roi_button.pack()

    def clamp(self, val, min_val, max_val):
        return max(min(val, max_val), min_val)

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.canvas.open_image(file_path)

    def toggle_roi_mode(self):
        self.roi_mode = not self.roi_mode
        self.roi_button.config(relief="sunken" if self.roi_mode else "raised")

        # Configure or remove canvas bindings
        if self.roi_mode:
            self.canvas.bind('<ButtonPress-1>', self.start_rect)
            self.canvas.bind('<B1-Motion>', self.update_rect)
            self.canvas.bind('<ButtonRelease-1>', self.fix_rect)
        else:
            self.canvas.bind('<ButtonPress-1>', self.canvas.drag_start)
            self.canvas.bind('<B1-Motion>', self.canvas.drag_move)
            self.canvas.unbind('<ButtonRelease-1>')

    def start_rect(self, event):
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
        self.start_x = self.clamp(self.canvas.canvasx(event.x), 0, self.canvas.width)
        self.start_y = self.clamp(self.canvas.canvasy(event.y), 0, self.canvas.height)
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y)
        #改变矩形颜色
        self.canvas.itemconfig(self.rect_id, outline='red')

    def update_rect(self, event):
        current_x = self.clamp(self.canvas.canvasx(event.x), 0, self.canvas.width)
        current_y = self.clamp(self.canvas.canvasy(event.y), 0, self.canvas.height)
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, current_x, current_y)

    def fix_rect(self, event):
        self.end_x = self.clamp(self.canvas.canvasx(event.x), 0, self.canvas.width)
        self.end_y = self.clamp(self.canvas.canvasy(event.y), 0, self.canvas.height)
        self.roi = [int(min(self.start_y, self.end_y) / self.canvas.scale), int(max(self.start_y, self.end_y) / self.canvas.scale), 
                    int(min(self.start_x, self.end_x) / self.canvas.scale), int(max(self.start_x, self.end_x) / self.canvas.scale)]
        print(f'ROI: {self.roi}')

    def save_roi(self):
        if self.roi:
            roi_str = str(self.roi)
            self.master.clipboard_clear()
            self.master.clipboard_append(roi_str)
            messagebox.showinfo("ROI", f"ROI coordinates saved to clipboard: {roi_str}")

if __name__ == '__main__':
    root = tk.Tk()
    root.geometry('800x600')
    editor = ROIEditor(root)
    root.mainloop()
