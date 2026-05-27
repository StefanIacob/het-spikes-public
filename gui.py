import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from skimage.draw import line_aa, disk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from collections import deque


def get_tk_im(array):
    array_im = np.asarray(array * 255, dtype='uint8')
    pilim = Image.fromarray(array_im)
    img = ImageTk.PhotoImage(image=pilim)
    return img


def draw_line(array_pos, array_neg, coord_start, coord_end, weight):
    """
    Draws a line on a np array.
    :param array: ndarray
        Image array to draw a line on.
    :param coord_start: (int, int)
        x and y position of start of line.
    :param coord_end: (int, int)
        x and y position of end of line.
    :return: ndarray
        Original image array with line drawn.
    """
    x1, y1 = coord_start
    x2, y2 = coord_end
    assert array_pos.shape[0] > x1 and array_pos.shape[0] > x2 and array_pos.shape[1] > y1 \
           and array_pos.shape[1] > y2, 'coordinates not in range '
    assert x1 >= 0 and x2 >= 0 and y1 >= 0 and y2 >= 0, 'Coordinates should be non-negative'
    assert array_pos.shape == array_neg.shape
    rr, cc, val = line_aa(x1, y1, x2, y2)
    if weight >= 0:
        array_pos[rr, cc] = 1 - val * weight * 2
    else:
        array_neg[rr, cc] = 1 - val * -weight * 2
    return array_pos, array_neg


class NetworkDrawer(object):

    def __init__(self, DDN):
        self.DDN = DDN
        x_range = self.DDN.x_range
        y_range = self.DDN.y_range
        width = max(x_range[1] - x_range[0], 0.01)
        height = max(y_range[1] - y_range[0], 0.01)
        window_width = 500
        self.spacing = window_width/width
        self.dot_size = 5
        self.w = int(window_width + 2*self.dot_size + 2)
        self.h = int(window_width * height/width + 2*self.dot_size + 2)
        grid = self.DDN.coordinates

        # shift and scale grid
        grid[:, 0] -= np.min(grid[:, 0])
        grid[:, 1] -= np.min(grid[:, 1])
        self.grid = grid

        self.base_drawing = self.draw_base()
        self.current_drawing = np.copy(self.base_drawing)


    def draw_base(self):
        pos, neg = self.get_connection_drawing()
        scale_bar = self.get_scalebar_drawing()
        connections_all = pos + neg

        dot_ex, dot_in = self.get_dots_drawing(True)
        connection_mask = np.stack([dot_in + dot_ex + scale_bar,
                          dot_in + dot_ex + scale_bar,
                          dot_in + dot_ex + scale_bar], axis=-1)

        img_c = np.stack([pos, connections_all, neg], axis=-1) * (connection_mask == 0)
        return img_c

    def get_image(self):
        return get_tk_im(self.current_drawing)

    def update_drawing(self, spike_vis=False):
        dot_ex, dot_in = self.get_dots_drawing(spike_vis=spike_vis)
        zer = np.zeros_like(dot_ex)
        dots_drawing = np.stack([dot_in, dot_ex, zer], axis=-1)
        img = self.base_drawing + dots_drawing
        self.current_drawing = img

    def get_connection_drawing(self):
        """
        Generates an image (w by h numpy array) of the non-zero connections in this network.
        :return: None
        """

        grid = self.grid 
        weights = self.DDN.get_total_weights()
        # scale the weights so they become visible
        weights = weights / np.std(weights)
        N = grid.shape[0]
        connections_pos = np.ones((self.w, self.h))
        connections_neg = np.ones((self.w, self.h))
        for n1 in range(N):
            for n2 in range(N):
                if weights[n1, n2] != 0:
                    x1 = int(grid[n1, 0] * self.spacing + self.dot_size)
                    y1 = int(grid[n1, 1] * self.spacing + self.dot_size)
                    x2 = int(grid[n2, 0] * self.spacing + self.dot_size)
                    y2 = int(grid[n2, 1] * self.spacing + self.dot_size)
                    connections_pos, connections_neg = draw_line(connections_pos, connections_neg, (x1, y1), (x2, y2),
                                                                 weights[n1, n2])
        return connections_pos, connections_neg

    def get_dots_drawing(self, init=False, spike_vis=False):
        """
        Generates an image as a ndarray with neuron activation drawn as dots according to coordinates attribute.
        :param init: bool
            Sets activity of all neurons to 1. Only used when initial connection base visualisation is being
            created.
        :return: ndarray, ndarray
            Two w by h array with excitatory and inhibitory neuron activation respectively as dots drawn in the
            array.
        """
        coordinates = self.grid
        if spike_vis:
            act = self.DDN.S[:, 0]
        else:
            act = self.DDN.A[:, 0] #* self.DDN.n_type
        if init:
            act = np.ones_like(act)
        N = coordinates.shape[0]
        dots_ex = np.zeros((self.w, self.h))
        dots_in = np.zeros((self.w, self.h))
        for i in range(N):
            a = act[i]
            x = np.round(coordinates[i, 0] * self.spacing + self.dot_size)
            y = np.round(coordinates[i, 1] * self.spacing + self.dot_size)
            rr, cc = disk((x, y), self.dot_size)

            if a > 0:
                dots_ex[rr, cc] = a
                dots_in[rr, cc] = 0
            else:
                dots_ex[rr, cc] = 0
                dots_in[rr, cc] = -a
        return dots_ex, dots_in

    def get_scalebar_drawing(self):
        # compute how many dts fit in a quarter screen
        width = self.h
        dt_width = self.DDN.dt * self.DDN.propagation_vel * self.spacing
        n_dt = int(np.floor(.45 * width/dt_width))

        scale_base = np.zeros((self.w, self.h))
        start_pos = np.array([self.w - 30, 20])
        end_pos = start_pos + np.array([0, n_dt * dt_width])
        # main line
        rr, cc, val = line_aa(int(start_pos[0]), int(start_pos[1]), int(end_pos[0]), int(end_pos[1]))
        scale_base[rr, cc] = val
        # vertical markers
        rr, cc, val = line_aa(int(start_pos[0] + 8), int(start_pos[1]), int(start_pos[0] - 8), int(start_pos[1]))
        scale_base[rr, cc] = val
        rr, cc, val = line_aa(int(end_pos[0] + 8), int(end_pos[1]), int(end_pos[0] - 8), int(end_pos[1]))
        scale_base[rr, cc] = val

        for i in range(n_dt):
            rr, cc, val = line_aa(int(start_pos[0] + 5), int(start_pos[1] + i * dt_width),
                                  int(start_pos[0] - 5), int(start_pos[1] + i * dt_width))
            scale_base[rr, cc] = val
        return scale_base


class ControlPanel(object):
    def __init__(self, tk_root, ddn, dim=1, window=300, external_input=None):
        self.dim = max(1, int(dim))
        self.root = tk.Toplevel(tk_root)
        self.root.title("Network Control Panel")
        self.root.protocol("WM_DELETE_WINDOW", self.destroy)
        self.DDN = ddn

        # --- State ---
        self.window = int(window)
        self.values = [deque(maxlen=self.window) for _ in range(self.dim)]
        sf = 4 #10e-1
        self.range_min = -1 * sf
        self.range_max = 1 * sf
        self._destroyed = False

        # --- Layout ---
        left = tk.Frame(self.root)
        left.grid(row=0, column=0, sticky="nsw", padx=6, pady=6)
        right = tk.Frame(self.root)
        right.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # --- Sliders ---
        slider_frame = tk.LabelFrame(left, text=f"Slider input ({self.dim}D)", padx=6, pady=6)
        slider_frame.pack(fill="x")

        self.scales = []
        self.value_labels = []
        for i in range(self.dim):
            row = tk.Frame(slider_frame)
            row.pack(fill="x", pady=(2 if i > 0 else 0, 2))

            tk.Label(row, text=f"Dim {i}").pack(side="left", padx=(0, 6))

            scale = tk.Scale(
                row, from_=0, to=100, resolution=1, length=220,
                orient="horizontal",
                command=lambda _=None, idx=i: self._update_value_label(idx)
            )
            scale.set(50)
            scale.pack(side="left", fill="x", expand=True)

            val_lbl = tk.Label(row, text="value: 0.0000")
            val_lbl.pack(side="left", padx=(6, 0))

            self.scales.append(scale)
            self.value_labels.append(val_lbl)

        # Global range controls
        range_frame = tk.Frame(slider_frame)
        range_frame.pack(fill="x", pady=(8, 5))
        tk.Label(range_frame, text="min").pack(side="left")
        self.min_entry = tk.Entry(range_frame, width=10)
        self.min_entry.insert(0, str(self.range_min))
        self.min_entry.pack(side="left", padx=4)
        tk.Label(range_frame, text="max").pack(side="left")
        self.max_entry = tk.Entry(range_frame, width=10)
        self.max_entry.insert(0, str(self.range_max))
        self.max_entry.pack(side="left", padx=4)
        tk.Button(slider_frame, text="Apply range", command=self._apply_range).pack(fill="x")

        # Bind Enter to apply range
        self.min_entry.bind("<Return>", lambda _e: self._apply_range())
        self.min_entry.bind("<KP_Enter>", lambda _e: self._apply_range())
        self.max_entry.bind("<Return>", lambda _e: self._apply_range())
        self.max_entry.bind("<KP_Enter>", lambda _e: self._apply_range())

        # Radiobuttons for switching between slider input and given input
        input_options = {"Manual Input": 0}
        if not (external_input is None):
            self.external_input = external_input
            self.current_input_index=0
            self.t_max = external_input.shape[1]
            input_options["External Input"] = 1  # If external option is given, extend the options dict
        rb_frame = tk.LabelFrame(left, text='Input type selection')
        rb_frame.pack(fill="x")
        self.selected_input_option = tk.IntVar()
        self.selected_input_option.set(0)
        for (name, val) in input_options.items():
            tk.Radiobutton(rb_frame, text=name, variable=self.selected_input_option, value=val).pack(fill='x')

        # Spike cost slider
        beta_frame = tk.LabelFrame(left, text='Spike cost')
        beta_frame.pack(fill="x")
        tk.Label(beta_frame, text="Spike cost").pack(side="left", padx=(0, 6))
        self.val_lbl_beta = tk.Label(beta_frame, text="value: 0.0000")
        self.val_lbl_beta.pack(side="left", padx=(6, 0))
        self.beta_scale = tk.Scale(
            beta_frame, from_=0, to=2, resolution=.01, length=220,
            orient="horizontal"
        )
        self.beta_scale.set(0)
        self.beta_scale.pack(side="left", fill="x", expand=True)

        tk.Button(beta_frame, text="Change spike cost", command=self._change_spike_cost).pack(fill="x")

        # Tau slider
        tau_frame = tk.LabelFrame(left, text='Time scale')
        tau_frame.pack(fill="x")
        tk.Label(tau_frame, text="Time scale").pack(side="left", padx=(0, 6))
        # self.val_lbl_beta = tk.Label(beta_frame, text="value: 0.0000")
        # self.val_lbl_beta.pack(side="left", padx=(6, 0))
        self.tau_scale = tk.Scale(
            tau_frame, from_=0.0001, to=.1, resolution=.00001, length=220,
            orient="horizontal"
        )
        self.tau_scale.set(0.01)
        self.tau_scale.pack(side="left", fill="x", expand=True)

        tk.Button(tau_frame, text="Change tau", command=self._change_tau_simple).pack(fill="x")

        # Knockout button
        ko_frame = tk.LabelFrame(left, text="Knockout experiment")
        ko_frame.pack(fill="x")
        tk.Label(ko_frame, text="Nr of neurons").pack(side="left")
        self.n_ko = tk.Entry(ko_frame, width=10)
        self.n_ko.pack(side="left", padx=4)
        tk.Button(ko_frame, text="Knockout neurons", command=self._knockout_callback).pack(fill="x")
        tk.Button(ko_frame, text="Reset neurons", command=self.DDN.reset_ko).pack(fill="x")
        # --- Single Matplotlib subplot with multiple lines ---
        self.fig = Figure(figsize=(6, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title(f"Slider values (last {self.window} steps)")
        self.ax.set_xlabel("step")
        self.ax.set_ylabel("value")
        self.ax.set_xlim(0, self.window)
        self.ax.set_ylim(self.range_min, self.range_max)

        # One line per dimension on the SAME axes
        self.lines = []
        for i in range(self.dim):
            (line,) = self.ax.plot([], [], label=f"Dim {i}")
            self.lines.append(line)
        if self.dim > 1:
            self.ax.legend(loc="upper right")

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Initialize labels
        for i in range(self.dim):
            self._update_value_label(i)

    # --------------------
    # Public API
    # --------------------
    def update(self):
        """Perform one GUI + plot update step. Returns False if window was closed."""
        # print(self.selected_input_option.get())
        if self._destroyed or not self.is_open():
            return False
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self._destroyed = True
            return False

        # Append current values
        v = self._current_values()
        for i in range(self.dim):
            self.values[i].append(v[i])

        # Update all lines on the same axes
        for i in range(self.dim):
            x = list(range(len(self.values[i])))
            y = list(self.values[i])
            self.lines[i].set_data(x, y)

        self.ax.set_xlim(0, self.window)
        self.canvas.draw_idle()

        # Update labels
        for i in range(self.dim):
            self._update_value_label(i)

        self._update_beta()

        return True

    def get_value(self) -> np.ndarray:
        """Return the current slider values as an array of shape (dim,)."""
        use_external_input = self.selected_input_option.get() == 1
        if use_external_input:
            external_i = self.external_input[:, self.current_input_index]
            if self.current_input_index == self.t_max - 1:
                self.current_input_index = 0  # reset input to start
            else:
                self.current_input_index += 1
            return external_i

        return self._current_values()

    def is_open(self):
        """True if the window still exists."""
        try:
            return bool(self.root.winfo_exists())
        except tk.TclError:
            return False

    def destroy(self):
        """Close and clean up the window."""
        if not self._destroyed:
            try:
                self.root.destroy()
            except tk.TclError:
                pass
            self._destroyed = True

    # --------------------
    # Internal helpers
    # --------------------
    def _parse_float(self, s: str) -> float:
        return float(s.strip().replace(",", "."))

    def _knockout_callback(self):
        self.DDN.reset_ko()
        try:
            n_neurons = int(self.n_ko.get())
        except Exception:
            n_neurons = 0
        N = self.DDN.N
        ko_indices = np.random.choice(N, replace=False, size=n_neurons)
        print(ko_indices)
        self.DDN.knock_out(ko_indices)



    def _apply_range(self):
        """Apply min/max to the single axes and update y-axis immediately."""
        try:
            new_min = self._parse_float(self.min_entry.get())
            new_max = self._parse_float(self.max_entry.get())
            if new_min >= new_max:
                raise ValueError("min must be < max")
        except Exception:
            self.min_entry.delete(0, tk.END)
            self.min_entry.insert(0, str(self.range_min))
            self.max_entry.delete(0, tk.END)
            self.max_entry.insert(0, str(self.range_max))
            return

        self.range_min, self.range_max = new_min, new_max
        self.ax.set_ylim(self.range_min, self.range_max)
        self.canvas.draw_idle()
        for i in range(self.dim):
            self._update_value_label(i)

    def _current_values(self) -> np.ndarray:
        span = (self.range_max - self.range_min)
        vals = []
        for scale in self.scales:
            pos = scale.get() / 100.0
            vals.append(self.range_min + pos * span)
        return np.array(vals, dtype=float)

    def _update_value_label(self, idx: int):
        v = self._current_values()[idx]
        self.value_labels[idx].config(text=f"value: {v:.4f}")

    def _update_beta(self):
        v = float(self.beta_scale.get())
        self.val_lbl_beta.config(text=str(v))


    def _change_spike_cost(self):
        new_beta = float(self.beta_scale.get())
        # reinitialize network
        self.DDN.change_spike_cost(new_beta)

    def _change_tau_simple(self):
        new_tau = float(self.tau_scale.get())
        self.DDN.change_tau(new_tau)


class DecoderPanel(object):

    def __init__(self, tk_root, ddn, control_panel, target_plant=None, trace_len=30):
        self.ddn = ddn
        assert not (ddn.decoder is None)
        self.decoder = ddn.decoder
        self.encoder = ddn.encoder
        self.control_panel = control_panel
        self.dim = self.decoder.D.shape[1]
        self.window = int(trace_len)

        self.x_decoded = np.zeros(shape=(self.dim, self.window))
        self.x_input = np.zeros(shape=(self.dim, self.window))
        self.target_plant = None
        if not(target_plant is None):
            self.target_plant = target_plant
        self.x_target = np.zeros(shape=(self.dim, self.window))


        # --- Tk + Matplotlib UI ---
        self.root = tk.Toplevel(tk_root)
        self.root.title("Decoder panel")
        # self.root.protocol("WM_DELETE_WINDOW", self.destroy)
        # === Top control frame ===
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=6, pady=4)


        if self.dim == 2:
            # Plot widget
            # --- Matplotlib plot
            self.fig = Figure(figsize=(5, 3), dpi=100)
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlabel("x1")
            self.ax.set_ylabel("x2")
            # self.ax.set_ylim()
            self.ax.set_title("Encoded signal")

            (self.line_decoded,) = self.ax.plot(self.x_decoded[0, :],self.x_decoded[1, :], label="Decoded signal")
            self.current_decoded = self.ax.scatter([self.x_decoded[0, -1]],[self.x_decoded[1, -1]])

            (self.line_input,) = self.ax.plot(self.x_input[0, :], self.x_input[1, :], label="input signal")
            self.current_input = self.ax.scatter([self.x_input[0, -1]], [self.x_input[1, -1]])

            (self.line_target,) = self.ax.plot(self.x_target[0, :], self.x_target[1, :], label="target signal")
            self.current_target = self.ax.scatter([self.x_target[0, -1]], [self.x_target[1, -1]])

            self.ax.legend(loc="upper right")
            self.canvas = FigureCanvasTkAgg(self.fig, master=top)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
            self.update()

    def update(self):
        estimate = self.decoder.get_estimate()
        input = self.control_panel.get_value()
        if self.target_plant is not None:
            target = self.target_plant.get_x()
        else:
            target = input

        # roll buffs
        self.x_decoded[:, :-1] = self.x_decoded[:, 1:]
        self.x_decoded[:, -1] = estimate

        self.x_input[:, :-1] = self.x_input[:, 1:]
        self.x_input[:, -1] = input

        self.x_target[:, :-1] = self.x_target[:, 1:]
        self.x_target[:, -1] = target

        self.traj_plot_2D()

        lim_min = self.control_panel.range_min * 2
        lim_max = self.control_panel.range_max * 2

        self.ax.set_xlim(lim_min, lim_max)
        self.ax.set_ylim(lim_min, lim_max)

        # lim_max = self.control_panel.range_max
        #
        # self.ax.set_xlim(-lim_max, lim_max)
        # self.ax.set_ylim(-lim_max, lim_max)

    def traj_plot_2D(self):
        x1 = self.x_decoded[0, :]
        x2 = self.x_decoded[1, :]

        self.line_decoded.set_xdata(x1)
        self.line_decoded.set_ydata(x2)
        self.current_decoded.set_offsets([[x1[-1], x2[-1]]])

        # input
        x1_i = self.x_input[0, :]
        x2_i = self.x_input[1, :]

        self.line_input.set_xdata(x1_i)
        self.line_input.set_ydata(x2_i)
        self.current_input.set_offsets([[x1_i[-1], x2_i[-1]]])

        # target
        x1_t = self.x_target[0, :]
        x2_t = self.x_target[1, :]

        self.line_target.set_xdata(x1_t)
        self.line_target.set_ydata(x2_t)
        self.current_target.set_offsets([[x1_t[-1], x2_t[-1]]])


        self.canvas.draw_idle()


class SingleNeuronProbe(object):
    def __init__(self, tk_root, ddn, neuron_index, window=300, title="Single Neuron Probe"):
        self.ddn = ddn
        self.neuron_index = neuron_index
        self.window = int(window)
        # --- Tk + Matplotlib UI ---
        self.root = tk.Toplevel(tk_root)
        self.root.title(f"{title}")
        self.root.protocol("WM_DELETE_WINDOW", self.destroy)
        self._destroyed = False

        # === Top control frame ===
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=6, pady=4)

        tk.Label(top, text=f"Neuron index (0–{self.ddn.N - 1}):").pack(side="left")

        # self.index_var = tk.StringVar(value=str(self.neuron_index))
        self.index_entry = tk.Entry(top, textvariable=self.neuron_index, width=6)
        self.index_entry.pack(side="left", padx=4)
        tk.Button(top, text="Set", command=self._apply_index).pack(side="left", padx=(4, 0))

        # Also apply when pressing Enter
        self.index_entry.bind("<Return>", lambda _e: self._apply_index())
        self.index_entry.bind("<KP_Enter>", lambda _e: self._apply_index())


        # --- rolling buffers ---
        self.V_hist = np.zeros(self.window, dtype=float)
        self.I_hist = np.zeros(self.window, dtype=float)
        self.R_hist = np.zeros(self.window, dtype=float)
        bl_thresh = ddn.T[neuron_index]
        self.T_hist = np.ones(self.window, dtype=float) * bl_thresh



        frame = tk.Frame(self.root)
        frame.pack(fill="both", expand=True)

        self.fig = Figure(figsize=(15, 6), dpi=100)
        self.axv = self.fig.add_subplot(311)
        self.axv.set_title(f"V (last {self.window} steps)")
        self.axv.set_xlabel("time step")
        self.axv.set_ylabel("V")

        self.axi = self.fig.add_subplot(312)
        self.axi.set_title(f"I (last {self.window} steps)")
        self.axi.set_xlabel("time step")
        self.axi.set_ylabel("I")
        self.axi.set_ylim(-5e-9, 10e-9)

        self.axr = self.fig.add_subplot(313)
        self.axr.set_title(f"Instantaneous rates (last {self.window} steps)")
        self.axr.set_xlabel("time step")
        self.axr.set_ylabel("r (Hz)")
        self.axr.set_ylim(0, np.max([5, np.max(self.R_hist)]))

        x = np.arange(self.window)
        (self.line_V,) = self.axv.plot(x, self.V_hist, label="V")
        (self.line_I,) = self.axi.plot(x, self.I_hist, label="I")
        (self.line_R,) = self.axr.plot(x, self.R_hist, label="r")
        (self.line_T,) = self.axv.plot(x, self.T_hist, linestyle="--", label="threshold")

        self.axv.set_xlim(0, self.window - 1)
        self._set_ylim_pad()

        self.axv.legend(loc="upper right")
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ---------------- Public API ----------------
    def update(self):
        """
        Advance one step: read ddn.V[neuron_index], shift buffer, redraw plot.
        Returns False if the window is closed.
        """
        if self._destroyed or not self.is_open():
            return False

        # Process Tk events once
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self._destroyed = True
            return False

        # ---- rolling buffer logic ----
        self.V_hist[:-1] = self.V_hist[1:]
        self.I_hist[:-1] = self.I_hist[1:]
        self.R_hist[:-1] = self.R_hist[1:]
        self.T_hist[:-1] = self.T_hist[1:]
        new_V = float(self.ddn.V[self.neuron_index])
        new_I = float(self.ddn.I[self.neuron_index])
        new_R = float(self.ddn.decoder.get_ratesHz()[self.neuron_index])
        new_T = float(self.ddn.T[self.neuron_index])

        self.V_hist[-1] = new_V
        self.I_hist[-1] = new_I
        self.R_hist[-1] = new_R
        self.T_hist[-1] = new_T

        # ---- redraw ----
        self.line_V.set_ydata(self.V_hist)
        self.line_T.set_ydata(self.T_hist)
        self.line_I.set_ydata(self.I_hist)
        self.line_R.set_ydata(self.R_hist)
        self.axr.set_ylim(0, np.max([5, np.max(self.R_hist)]))
        self._set_ylim_pad()
        self.canvas.draw_idle()

        return True

    def is_open(self):
        try:
            return bool(self.root.winfo_exists())
        except tk.TclError:
            return False

    def destroy(self):
        if not self._destroyed:
            try:
                self.root.destroy()
            except tk.TclError:
                pass
            self._destroyed = True

    # ---------------- Internals ----------------
    def _apply_index(self):
        """Read index from entry, validate, and update neuron selection."""
        try:
            new_index = int(self.index_entry.get())
            # print(new_index)
            if not (0 <= new_index < self.ddn.N):
                raise ValueError
        except Exception:
            # Revert to current index on invalid input
            # self.index_var.set(str(self.neuron_index))
            return

        self.neuron_index = new_index
        # self.axv.set_title(f"Neuron {self.neuron_index} V (last {self.window} steps)")
        # self.canvas.draw_idle()

    def _set_ylim_pad(self):
        """
        Set y-limits with a small pad that covers both V and threshold.
        """
        ymin = float(np.min([self.V_hist.min(), self.T_hist.min()]))
        ymax = float(np.max([self.V_hist.max(), self.T_hist.max()]))
        if ymin == ymax:  # avoid zero-height axis
            pad = 1.0 if ymin == 0 else abs(ymin) * 0.1
            ymin -= pad
            ymax += pad
        else:
            rng = ymax - ymin
            pad = 0.05 * rng
            ymin -= pad
            ymax += pad
        self.axv.set_ylim(ymin, ymax)


class RasterPlot(object):

    def __init__(self, tk_root, ddn):
        self.ddn = ddn
        buffersize = ddn.B

        # --- Tk + Matplotlib UI ---
        self.root = tk.Toplevel(tk_root)
        self.root.title("Rasterplot")

        # === Top control frame ===
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=6, pady=4)

        # Plot widget
        # --- Matplotlib plot
        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel(f"Time ({'arb.' if ddn.dt == 1.0 else ''} units)")
        self.ax.set_ylabel("Neuron")
        self.ax.set_ylim(-0.5, ddn.N - 0.5)  # nice tight y-lims
        self.ax.set_title("Spike raster")
        self.canvas = FigureCanvasTkAgg(self.fig, master=top)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.update()

    def _spikes_to_times(self, spikes, dt):
        N, T_disc = spikes.shape
        T_cont = (np.arange(T_disc) + 0.5) * dt
        # convert each row to an array of spike times
        spike_times_per_neuron = [T_cont[spikes[i].astype(bool)] for i in range(N)]
        return T_cont, spike_times_per_neuron

    def update(self):
        spikes = self.ddn.S
        # dt = self.ddn.dt
        # T_cont, spike_times_per_neuron = self._spikes_to_times(spikes, dt)
        # self.ax.eventplot(spike_times_per_neuron, linelengths=0.8, linewidths=0.6)
        self.ax.imshow(spikes)
        self.canvas.draw_idle()

class NetworkPanel(object):
    def __init__(self, tk_root, ddn, spike_vis):
        self.net_drawer = NetworkDrawer(ddn)
        self.root = tk_root
        self.network_frame = tk.Frame(self.root)
        self.network_frame.pack(side=tk.LEFT)
        self.network_canvas = tk.Canvas(self.network_frame, width=self.net_drawer.h + 100,
                                        height=self.net_drawer.w + 100)
        self.network_canvas.grid(row=0, column=0)
        self.spike_vis = spike_vis

    def update(self):
        self.net_drawer.update_drawing(spike_vis=self.spike_vis)
        self.net_img = self.net_drawer.get_image()
        self.network_canvas.create_image(5, 5, anchor="nw", image=self.net_img)
        self.root.update()

class DistDelayGUI(object):
    """
    Animation for the distance-based delay network
    """

    def __init__(self, dist_delay_net, plant=None, inputs=None, probe_index=None, use_rasterplot=False,
                 in_size=1, spike_vis=False):
        self.DDN = dist_delay_net
        self.use_rasterplot = use_rasterplot
        self.use_plant = False
        self.root = tk.Tk()
        self.ddn_panel = NetworkPanel(self.root, dist_delay_net, spike_vis=spike_vis)
        self.control_panel = None
        self.gui_net_input = None
        self.control_panel = ControlPanel(self.root, dist_delay_net, dim=in_size, external_input=inputs)
        self.rasterplot_panel = None
        if use_rasterplot:
            self.rasterplot_panel = RasterPlot(self.root, dist_delay_net)
        self.single_probe = None
        if probe_index is not None:
            self.single_probe = SingleNeuronProbe(self.root, dist_delay_net, probe_index)

        self.plant = None
        if not (plant is None):
            self.use_plant = True
            self.plant = plant

        self.use_decoder = False

        if dist_delay_net.decoder is not None:
            self.decoder_panel = DecoderPanel(self.root, dist_delay_net, self.control_panel, target_plant=self.plant)
            self.use_decoder = True


    def run(self):
        self._update_a()
        self.root.mainloop()

    def DDN_update(self, beta=None):
        inp = self.gui_net_input
        self.DDN.update_step(inp)

    def _update_a(self):
        self.ddn_panel.update()

        if self.single_probe is not None:
            self.single_probe.update()
        if self.use_rasterplot:
            self.rasterplot_panel.update()
        if self.use_decoder:
            self.decoder_panel.update()

        net_inp = self.control_panel.get_value()
        self.control_panel.update()
        self.gui_net_input = net_inp
        if self.use_plant:
            self.plant.update(net_inp)

        self.DDN_update()

        self.root.after(0, self._update_a)

    def close(self):
        self.root.destroy()


class EvolutionGui(object):

    def __init__(self, image_list):

        self.dot_size = 5
        self.root = tk.Tk()

        self.frame1 = tk.Frame(self.root)
        self.frame1.pack(side=tk.LEFT)

        self.frame2 = tk.Frame(self.root)
        self.frame2.pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(self.frame1, width=600, height=600)
        self.canvas.grid(row=0, column=0)

        self.slider = tk.Scale(self.frame2, from_=0, to=len(image_list)-1, length=500)
        self.slider.grid(row=0, column=1)

        self.image_list = [get_tk_im(img) for img in image_list]

    def update(self):
        v = self.slider.get()
        img = self.image_list[v]
        self.canvas.delete('all')
        self.canvas.create_image(5, 5, anchor="nw", image=img)
        self.root.update()


def draw_scale(w, h, dt, spacing):
    # compute how many dts fit in a quarter screen
    width = h
    dt_width = dt * config.propagation_vel * spacing
    n_dt = int(np.floor(.45 * width/dt_width))

    scale_base = np.zeros((w, h))
    start_pos = np.array([w - 30, 20])
    end_pos = start_pos + np.array([0, n_dt * dt_width])
    # main line
    rr, cc, val = line_aa(int(start_pos[0]), int(start_pos[1]), int(end_pos[0]), int(end_pos[1]))
    scale_base[rr, cc] = val
    # vertical markers
    rr, cc, val = line_aa(int(start_pos[0] + 8), int(start_pos[1]), int(start_pos[0] - 8), int(start_pos[1]))
    scale_base[rr, cc] = val
    rr, cc, val = line_aa(int(end_pos[0] + 8), int(end_pos[1]), int(end_pos[0] - 8), int(end_pos[1]))
    scale_base[rr, cc] = val

    for i in range(n_dt):
        rr, cc, val = line_aa(int(start_pos[0] + 5), int(start_pos[1] + i * dt_width),
                              int(start_pos[0] - 5), int(start_pos[1] + i * dt_width))
        scale_base[rr, cc] = val
    return scale_base


def grid2dots_simple(coordinates, w, h, spacing, dot_size=5):

    N = coordinates.shape[0]
    dots_ex = np.zeros((w, h))
    for i in range(N):
        x = np.round(coordinates[i, 0] * spacing + dot_size)
        y = np.round(coordinates[i, 1] * spacing + dot_size)
        rr, cc = disk((x, y), dot_size)

        dots_ex[rr, cc] = 1
    return dots_ex


def grid2dots(coordinates, DDN, w, h, spacing, dot_size=5):

    act = DDN.A[:, 0] * DDN.n_type
    act = np.ones_like(act) * DDN.n_type
    N = coordinates.shape[0]
    dots_ex = np.zeros((w, h))
    dots_in = np.zeros((w, h))
    dots_input = np.zeros((w, h))
    for i in range(N):
        a = act[i]
        x = np.round(coordinates[i, 0] * spacing + dot_size)
        y = np.round(coordinates[i, 1] * spacing + dot_size)
        rr, cc = disk((x, y), dot_size)

        if i > DDN.size_in:
            if a > 0:
                dots_ex[rr, cc] = a
                dots_in[rr, cc] = 0
            else:
                dots_ex[rr, cc] = 0
                dots_in[rr, cc] = -a
        else:
            dots_input[rr, cc] = a
    return dots_ex, dots_in, dots_input


def get_connection_base(weights, grid, spacing, w, h, dot_size=5):

    # weights = np.asarray(weights > 0, dtype='uint8')
    N = grid.shape[0]
    connections_pos = np.ones((w, h))
    connections_neg = np.ones((w, h))

    for n1 in range(N):
        for n2 in range(N):
            if weights[n1, n2] != 0:
                x1 = int(grid[n1, 0] * spacing + dot_size)
                y1 = int(grid[n1, 1] * spacing + dot_size)
                x2 = int(grid[n2, 0] * spacing + dot_size)
                y2 = int(grid[n2, 1] * spacing + dot_size)
                connections_pos, connections_neg = draw_line(connections_pos, connections_neg, (x1, y1), (x2, y2),
                                                             weights[n1, n2])
                # xa1 = x2 + 0.95 * (x1 - x2) + 5
                # ya1 = y2 + 0.95 * (y1 - y2) + 5
                # xa2 = x2 + 0.95 * (x1 - x2) - 5
                # ya2 = y2 + 0.95 * (y1 - y2) - 5
                # connections = draw_line(connections, (int(xa1), int(ya1)), (x2, y2))
                # connections = draw_line(connections, (int(xa2), int(ya2)), (x2, y2))
    return connections_pos, connections_neg


def get_network_im(net, dot_size=5):
    grid = net.coordinates
    # shift and scale grid
    grid[:, 0] -= np.min(grid[:, 0])
    grid[:, 1] -= np.min(grid[:, 1])
    width = net.x_range[1] - net.x_range[0]
    height = net.y_range[1] - net.y_range[0]
    ratio = width / height
    if ratio > 1:
        w = 500
        h = int(w / ratio)
    else:
        h = 500
        w = int(h * ratio)

    spacing = w / width
    w += 2*dot_size + 2
    h += 2*dot_size + 2
    connections_pos, connections_neg = get_connection_base(net.W, grid, spacing, w, h, dot_size)
    dots_ex, dots_in, dots_input = grid2dots(grid, net, w, h, spacing, dot_size)
    scale_base = draw_scale(w, h, net.dt, spacing)
    all_d = np.stack([dots_in + dots_ex + dots_input + scale_base,
                      dots_in + dots_ex + scale_base,
                      dots_in + dots_ex + scale_base], axis=-1)
    connections_all = connections_pos + connections_neg

    img_c = np.stack([connections_pos, connections_all, connections_neg], axis=-1) * \
            (all_d == 0)

    zer = np.zeros_like(connections_pos)
    img_d = np.stack([dots_in, dots_ex, zer], axis=-1)
    img = img_c + img_d
    # img = get_tk_im(img)
    return img
