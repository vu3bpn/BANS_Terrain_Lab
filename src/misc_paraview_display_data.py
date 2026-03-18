"""
ParaView Macro: LAS RGB Point Cloud Viewer
==========================================
Install as a macro:
  Macros > Add New Macro > select this file
  Then run it from the Macros menu anytime.

A file-picker dialog will appear — select your .las or .laz file.
The point cloud will be displayed coloured by Red, Green, Blue arrays.
"""

from paraview.simple import (
    LASReader, GetActiveViewOrCreate, Show, Hide,
    ColorBy, GetColorTransferFunction, ResetCamera,
    Render, Calculator
)
import paraview.simple as pvs
from paraview import servermanager

# ── USER SETTINGS ──────────────────────────────────────────────────────────────
POINT_SIZE       = 2        # Rendered point size in pixels
SIXTEEN_BIT_RGB  = True     # True = 16-bit LAS (0-65535) | False = 8-bit (0-255)
BACKGROUND_COLOR = [0.08, 0.08, 0.08]  # Near-black background

# ── FILE PICKER ────────────────────────────────────────────────────────────────
try:
    # Qt file dialog (works in ParaView GUI)
    from paraview.qt import QApplication
    from PyQt5.QtWidgets import QFileDialog, QMessageBox
    import os

    app = QApplication.instance()

    las_file, _ = QFileDialog.getOpenFileName(
        None,
        "Open LAS / LAZ Point Cloud",
        "",
        "LAS/LAZ Files (*.las *.laz);;All Files (*.*)"
    )

    if not las_file:
        # User cancelled
        msg = QMessageBox()
        msg.setWindowTitle("LAS RGB Viewer")
        msg.setText("No file selected. Macro cancelled.")
        msg.exec_()
        raise SystemExit("Cancelled by user.")

except ImportError:
    # Fallback for older ParaView / non-Qt builds
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    las_file = filedialog.askopenfilename(
        title="Open LAS / LAZ Point Cloud",
        filetypes=[("LAS/LAZ files", "*.las *.laz"), ("All files", "*.*")]
    )
    root.destroy()

    if not las_file:
        raise SystemExit("Cancelled by user.")

print(f"[LAS Viewer] Loading: {las_file}")

# ── LOAD THE LAS FILE ──────────────────────────────────────────────────────────
reader = LASReader(FileName=las_file)
reader.UpdatePipeline()

# Inspect available arrays
data_info  = reader.GetDataInformation()
pd_info    = data_info.GetPointDataInformation()
arrays     = [pd_info.GetArrayInformation(i).GetName()
              for i in range(pd_info.GetNumberOfArrays())]

num_points = data_info.GetNumberOfPoints()
print(f"[LAS Viewer] Points: {num_points:,}")
print(f"[LAS Viewer] Arrays: {arrays}")

has_rgb = {"Red", "Green", "Blue"}.issubset(set(arrays))

# ── SET UP VIEW ────────────────────────────────────────────────────────────────
view = GetActiveViewOrCreate("RenderView")
view.Background           = BACKGROUND_COLOR
view.UseGradientBackground = 0

# ── HELPER: configure display as a point cloud ────────────────────────────────
def make_point_display(source):
    disp = Show(source, view)
    disp.Representation = "Point Gaussian"
    disp.GaussianRadius  = 0.0
    disp.ShaderPreset    = "Plain circle"
    disp.PointSize       = POINT_SIZE
    return disp

# ── COLOUR BY RGB ──────────────────────────────────────────────────────────────
if has_rgb:
    scale = 1.0 / (65535.0 if SIXTEEN_BIT_RGB else 255.0)

    calc = Calculator(Input=reader)
    calc.AttributeType   = "Point Data"
    calc.ResultArrayName = "RGB"
    calc.Function = (
        f"({scale}*Red)*iHat + ({scale}*Green)*jHat + ({scale}*Blue)*kHat"
    )
    calc.UpdatePipeline()

    Hide(reader, view)
    display = make_point_display(calc)

    # Direct RGB colour mapping — no LUT
    display.ColorArrayName       = ["POINTS", "RGB"]
    display.MapScalars           = 0
    display.MultiComponentsMapping = 0

    display.SetScalarBarVisibility(view, False)
    print("[LAS Viewer] Colouring by RGB.")

# ── FALL BACK: colour by elevation ────────────────────────────────────────────
else:
    print("[LAS Viewer] WARNING: Red/Green/Blue arrays not found.")
    print("[LAS Viewer] Falling back to elevation colouring.")

    display = make_point_display(reader)
    ColorBy(display, ("POINTS", "Points"), component=2)
    lut = GetColorTransferFunction("Points")
    lut.ApplyPreset("Cool to Warm", True)
    display.SetScalarBarVisibility(view, True)

# ── RENDER ─────────────────────────────────────────────────────────────────────
ResetCamera(view)
Render()

import os
print(f"[LAS Viewer] Done — '{os.path.basename(las_file)}' loaded successfully.")

# ── NOTIFY USER (Qt popup) ─────────────────────────────────────────────────────
try:
    from PyQt5.QtWidgets import QMessageBox
    mode = "True RGB colour" if has_rgb else "Elevation colour (no RGB arrays found)"
    info = QMessageBox()
    info.setWindowTitle("LAS RGB Viewer")
    info.setText(
        f"<b>{os.path.basename(las_file)}</b><br>"
        f"Points: <b>{num_points:,}</b><br>"
        f"Mode: <b>{mode}</b>"
    )
    info.setIcon(QMessageBox.Information)
    info.exec_()
except Exception:
    pass  # silently skip popup if Qt unavailable