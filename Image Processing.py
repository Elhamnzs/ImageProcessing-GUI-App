import sys, time, pdb
import cv2  # pip install opencv-python
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi  # pip install scipy
import scipy.signal as sig
from scipy.ndimage import gaussian_filter

from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, pyqtSlot, QRect, QPoint, QSize
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QWidget, QMessageBox, QVBoxLayout, QLabel, QSlider, QPushButton, QRubberBand

from S3GUI1 import Ui_MainWindow
from GammaWindow import Ui_Form as Ui_FormGamma
import resources2_rc
from GaussianBlurWindow import Ui_GaussianBlurWindow


from PyQt5.QtCore import QThread, pyqtSignal

# GaussianBlurWindow.py

from scipy.ndimage import gaussian_filter
import numpy as np
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtGui
from GaussianBlurWindow import Ui_GaussianBlurWindow  # Ensure correct import path



class GaussianBlurWindow(QWidget):
    def __init__(self, parent=None):
        super(GaussianBlurWindow, self).__init__()
        self.parent = parent
        self.p_ui = Ui_GaussianBlurWindow()
        self.p_ui.setupUi(self)

        # Connect slider and apply button
        self.p_ui.sliderSigma.valueChanged.connect(self.SliderSigma)
        self.p_ui.btnApply.clicked.connect(self.apply_blur)

        # Display the original image when the window opens
        self.displayImage(image=self.parent.out_image, image2=self.parent.out_image)
        self.show()

    def displayImage(self, image=None, image2=None):
        """Display the original and blurred images."""
        if image is not None and image.any():
            # Display original image
            h, w, ch = image.shape
            bytes_per_line = ch * w
            qformat = QImage.Format_RGB888 if ch == 3 else QImage.Format_RGBA8888
            qt_image = QImage(image.data, w, h, bytes_per_line, qformat)
            p = qt_image.scaled(self.p_ui.lblPreview.width(), self.p_ui.lblPreview.height(),
                                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.p_ui.lblPreview.setPixmap(QPixmap.fromImage(p))

        if image2 is not None and image2.any():
            # Display blurred (final) image
            h, w = image2.shape[:2]
            qt_image2 = QImage(image2.data, w, h, image2.strides[0], QImage.Format_Grayscale8)
            p2 = qt_image2.scaled(self.p_ui.lblPreview2.width(), self.p_ui.lblPreview2.height(),
                                  Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.p_ui.lblPreview2.setPixmap(QPixmap.fromImage(p2))

    def SliderSigma(self, sigma_value):
        """Adjusts the preview of the Gaussian blur based on slider value."""
        sigma = sigma_value * 0.1
        grayscale_image = np.mean(self.parent.out_image, axis=2).astype(np.uint8)
        blurred_image = gaussian_filter(grayscale_image, sigma=sigma)
        self.displayImage(image=self.parent.out_image, image2=blurred_image)

    def apply_blur(self):
        """Apply Gaussian blur and send the blurred image back to the main window."""
        grayscale_image = np.mean(self.parent.out_image, axis=2).astype(np.uint8)
        sigma_value = self.p_ui.sliderSigma.value() * 0.1
        final_blurred_image = gaussian_filter(grayscale_image, sigma=sigma_value)
        self.parent.apply_blurred_image(final_blurred_image)
        self.close()



class EdgeDetectionThread(QThread):
    edge_detected = pyqtSignal(np.ndarray)  # Signal to emit the edge-detected image

    def __init__(self, image):
        super().__init__()
        self.image = image

    def run(self):
        # Convert to grayscale if the image is not already grayscale
        if len(self.image.shape) == 3:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = self.image

        # Apply the Canny Edge Detection
        edges = cv2.Canny(gray_image, 100, 200)  # Adjust thresholds as needed

        # Emit the result
        self.edge_detected.emit(edges)




class GammaWindow(QWidget):
    def __init__(self, parent):
        super(GammaWindow, self).__init__()
        self.parent = parent
        QWidget.__init__(self)
        Ui_FormGamma.__init__(self)

        self.p_ui = Ui_FormGamma()
        self.p_ui.setupUi(self)

        self.p_ui.slider_gamma.valueChanged.connect(self.SliderGamma)
        self.p_ui.pbValidateGamma.clicked.connect(self.updateMainWindow)

        self.displayImage(image=self.parent.out_image, image2=self.parent.out_image)
        self.show()

        self.bGammaChanged = False

    def displayImage(self, image=None, image2=None):
        if image.any():
            # Check the type of a variable, syntax is: if isinstance(variable, type):
            if isinstance(image, np.ndarray):
                print("Your image is an array (numpy or OpenCV)")
                # Convert to QImage
                if len(image.shape) == 3:
                    # Color images with alpha channel
                    if (image.shape[2]) == 4:
                        qformat = QImage.Format_RGBA8888
                    # Color images
                    else:
                        qformat = QImage.Format_RGB888
                    h, w, ch = image.shape
                    bytes_per_line = ch * w  # channel * width, also known as strides
                    self.qt_image = QtGui.QImage(image.data, w, h, bytes_per_line, qformat)
                else:
                    # Grayscale images
                    qformat = QImage.Format_Indexed8
                    self.qt_image = QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)

                # Convert QImage to QPixmap (format for labels)
                p = self.qt_image.scaled(self.p_ui.lblPreGamma.width(), self.p_ui.lblPreGamma.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.pix_image = QPixmap.fromImage(p)
                # Display image in the label
                self.p_ui.lblPreGamma.setPixmap(self.pix_image)

        if image2.any():
            # Check the type of a variable, syntax is: if isinstance(variable, type):
            if isinstance(image2, np.ndarray):
                print("Your image 2 is an array (numpy or OpenCV)")
                # Convert to QImage
                if len(image2.shape) == 3:
                    # Color images with alpha channel
                    if (image2.shape[2]) == 4:
                        qformat = QImage.Format_RGBA8888
                    # Color images
                    else:
                        qformat = QImage.Format_RGB888
                    h, w, ch = image2.shape
                    bytes_per_line = ch * w  # channel * width, also known as strides
                    self.qt_image = QtGui.QImage(image2.data, w, h, bytes_per_line, qformat)
                else:
                    # Grayscale images
                    qformat = QImage.Format_Indexed8
                    self.qt_image = QImage(image2, image2.shape[1], image2.shape[0], image2.strides[0], qformat)

                # Convert QImage to QPixmap (format for labels)
                p2 = self.qt_image.scaled(self.p_ui.lblPreGamma.width(), self.p_ui.lblPreGamma.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.pix_image2 = QPixmap.fromImage(p2)
                # Display image in the label
                self.p_ui.lblPostGamma.setPixmap(self.pix_image2)

    def SliderGamma(self, gamma):
        gamma = gamma * 0.1
        if gamma == 0:
            gamma = 0.0001
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        self.newimage = cv2.LUT(self.parent.out_image, table)
        self.bGammaChanged = True
        self.displayImage(image=self.parent.out_image, image2=self.newimage)

    def updateMainWindow(self):
        self.parent.DisplayImage(image=self.newimage)
        self.parent.out_image = self.newimage
        self.parent.add_to_history(self.out_image)
        self.close()


class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        QMainWindow.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.setWindowTitle("Image Processing Application")

        # --- Colors/Design
        self.setStyleSheet('QMainWindow { background-color: #333; color: white; }'
                           'QMenuBar { background-color: #333; color: white; }'
                           'QMenuBar:item:selected { background: #05F; color: white; }'
                           'QMenuBar#QAction { background-color: #333; color: white; }'
                           'QToolBar { background-color: #333; color: white; }'
                           'QLabel { background-color: #333; color: white; }'
                           'QGroupBox { background-color: #333; color: white; }')

        # Undo/Redo related attributes


        self.history = []
        self.history_index = -1  # Start with -1, meaning no image loaded yet

        # Initialize scale factor
        self.scale_factor = 1.0  # Default zoom level

        # Connect actions
        self.ui.actionZoom_in.triggered.connect(self.zoom_in)
        self.ui.actionZoom_out.triggered.connect(self.zoom_out)

        # Objects
        self.ui.actionOpen.triggered.connect(self.openImage)
        self.ui.actionSave.triggered.connect(self.saveImage)
        self.ui.actionPrint.triggered.connect(self.printImage)
        self.ui.actionUndo.triggered.connect(self.undoChanges)
        self.ui.actionRedo.triggered.connect(self.redoChanges)
        self.ui.actionUndo.setEnabled(False)
        self.ui.actionRedo.setEnabled(False)
        self.ui.actionCrop.triggered.connect(self.enable_crop_mode)
        self.ui.actionGrayscale.triggered.connect(self.convGrayscale)
        self.ui.actionRGB_to_BGR.triggered.connect(self.convBGR)
        self.ui.actionBGR_to_RGB.triggered.connect(self.convRGB)
        self.ui.actionNegative.triggered.connect(self.convNegative)
        self.ui.actionGaussian.triggered.connect(self.open_gaussian_blur_window)
        self.ui.actionMedian.triggered.connect(self.medianFilter)
        self.ui.actionAbout.triggered.connect(self.aboutBox)
        self.ui.horizontalSlider.valueChanged.connect(self.gammaSlider)

        self.ui.actionSharpening.triggered.connect(self.apply_sharpening)
        self.ui.actionSepia.triggered.connect(self.apply_sepia)

        self.ui.actionGamma.triggered.connect(self.showGammaWindow)

        # Image variables
        self.bgr_cv_image = 0       # Image after reading it with OpenCV, BGR
        self.cp_bgr_cv_image = 0  # Copy of the real input image
        self.rgb_cv_image = 0       # Image after converting it with OpenCV to RGB
        self.qt_image = 0           # Image after converting it to QT
        self.pix_image = 0          # Image after converting it to Pixmap
        self.out_image = 0          # The variable we will always use to make changes to the image
        self.bImgloaded = False     # For the resizeEvent

        # Hide second label (old output)
        self.ui.lblOutputImg.hide()
        self.ui.label_3.hide()
        self.ui.label_4.hide()
        self.ui.groupBox_2.hide()

        # To be able to use events, linked to: def eventFilter
        self.installEventFilter(self)
        self.ui.actionEdgeDetection.triggered.connect(self.start_edge_detection)

        # For cropping
        self.ui.actionCrop.setEnabled(True)
        self.rubber_band = None
        self.origin = QPoint()
        self.crop_rect = QRect()
        # Show main window
        self.show()

        # Initialize other elements
        self.bImgloaded = False
        self.display_scale_factor = 1.0  # Initialize separate zoom factor for display






    def enable_crop_mode(self):
        """Enable crop mode to allow user to draw a selection on the image."""
        QMessageBox.information(self, "Crop Mode", "Draw a box around the area you want to crop.")
        self.ui.lblInputImg.setCursor(Qt.CrossCursor)

    def mousePressEvent(self, event):
        """Start the rubber band selection when mouse is pressed."""
        if self.ui.lblInputImg.underMouse():
            # Convert to coordinates within lblInputImg
            self.origin = self.ui.lblInputImg.mapFromGlobal(event.globalPos())
            if not self.rubber_band:
                self.rubber_band = QRubberBand(QRubberBand.Rectangle, self.ui.lblInputImg)
            self.rubber_band.setGeometry(QRect(self.origin, QSize()))
            self.rubber_band.show()

    def mouseMoveEvent(self, event):
        """Update rubber band selection box as mouse moves."""
        if self.rubber_band and self.ui.lblInputImg.underMouse():
            # Convert to coordinates within lblInputImg
            end_point = self.ui.lblInputImg.mapFromGlobal(event.globalPos())
            self.rubber_band.setGeometry(QRect(self.origin, end_point).normalized())

    def mouseReleaseEvent(self, event):
        """Complete the selection and capture the crop area when mouse is released."""
        if self.rubber_band:
            self.crop_rect = self.rubber_band.geometry()
            self.rubber_band.hide()
            self.crop_image()

    def crop_image(self):
        """Crop the selected area from the image."""
        try:
            if not self.out_image.any():
                QMessageBox.warning(self, "No Image", "Please load an image first.")
                return

            x, y, w, h = self.crop_rect.getRect()
            cropped_image = self.out_image[y:y + h, x:x + w]

            if cropped_image.size == 0:
                QMessageBox.warning(self, "Invalid Crop", "Please select a valid crop area.")
                return

            # Ensure cropped image is correctly formatted for display
            cropped_image = np.ascontiguousarray(cropped_image)  # Ensure memory alignment for QImage
            if cropped_image.ndim == 3:  # For color images
                height, width, channel = cropped_image.shape
                qimage = QImage(cropped_image.data, width, height, width * channel, QImage.Format_RGB888)
            else:  # For grayscale images
                height, width = cropped_image.shape
                qimage = QImage(cropped_image.data, width, height, width, QImage.Format_Grayscale8)

            # Update main display with cropped image
            self.out_image = cropped_image
            self.DisplayImage(self.out_image)
            self.add_to_history(self.out_image)  # Save to history for undo/redo

            # Reset the cursor and crop rectangle
            self.ui.lblInputImg.setCursor(Qt.ArrowCursor)
            self.crop_rect = QRect()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while cropping: {e}")

    def open_gaussian_blur_window(self):
        """Open Gaussian blur window to apply the filter."""
        if self.out_image is not None:
            try:
                self.gaussian_blur_window = GaussianBlurWindow(parent=self)
                self.gaussian_blur_window.show()
            except Exception as e:
                print(f"Error opening Gaussian blur window: {e}")

    def apply_blurred_image(self, blurred_image):
        """Receive the blurred image from GaussianBlurWindow and update the main display."""
        self.out_image = blurred_image
        self.DisplayImage(self.out_image)
        self.add_to_history(self.out_image)


    def zoom_in(self):
        """Zoom in and add the zoomed image to history."""
        self.scale_factor *= 1.25  # Increase the zoom factor by 25%
        self.apply_zoom(add_to_history=True)

    def zoom_out(self):
        """Zoom out and add the zoomed image to history."""
        self.scale_factor /= 1.25  # Decrease the zoom factor by 25%
        self.apply_zoom(add_to_history=True)

    def apply_zoom(self, add_to_history=False):
        """Apply zoom based on scale factor and update the display."""
        if self.out_image is not None:
            # Resize for display only
            height, width = self.out_image.shape[:2]
            new_size = (int(width * self.scale_factor), int(height * self.scale_factor))
            zoomed_image = cv2.resize(self.out_image, new_size, interpolation=cv2.INTER_LINEAR)

            # Display the resized image
            self.DisplayImage(image=zoomed_image)

            # Add to history if specified
            if add_to_history:
                self.add_to_history(zoomed_image)

    def add_to_history(self, image):
        """Add the modified image to history."""
        if image is not None:
            # Clear any forward history if undo was used
            if self.history_index < len(self.history) - 1:
                self.history = self.history[:self.history_index + 1]

            # Add the current image to history and update the index
            self.history.append(image.copy())
            self.history_index += 1
            self.update_undo_redo_buttons()

    def undoChanges(self):
        """Undo the last change."""
        if self.history_index > 0:
            self.history_index -= 1
            self.out_image = self.history[self.history_index]
            self.scale_factor = 1.0  # Reset zoom level for undo
            self.DisplayImage(image=self.out_image)
            self.update_undo_redo_buttons()

    def redoChanges(self):
        """Redo the next change in history."""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.out_image = self.history[self.history_index]
            self.scale_factor = 1.0  # Reset zoom level for redo
            self.DisplayImage(image=self.out_image)
            self.update_undo_redo_buttons()

    def update_undo_redo_buttons(self):
        """Enable or disable undo/redo buttons based on history index."""
        self.ui.actionUndo.setEnabled(self.history_index > 0)
        self.ui.actionRedo.setEnabled(self.history_index < len(self.history) - 1)

    def openImage(self):
        """Load and display a new image."""
        filename, _ = QFileDialog.getOpenFileName(self, 'Open File', '', "Image Files (*.jpg *.jpeg *.png *.bmp)")
        if filename:
            bgr_cv_image = cv2.imread(filename)
            if bgr_cv_image is None:
                QMessageBox.warning(self, "Open Image", "Could not open image file.")
                return

            # Convert BGR image to RGB
            self.rgb_cv_image = cv2.cvtColor(bgr_cv_image, cv2.COLOR_BGR2RGB)
            self.out_image = self.rgb_cv_image
            self.DisplayImage(image=self.out_image)

            # Reset history on new load
            self.history = [self.out_image.copy()]
            self.history_index = 0
            self.update_undo_redo_buttons()  # Enable buttons appropriately
            print("[DEBUG] Image loaded and added to history.")


    def saveImage(self):
        filename, ext = QFileDialog.getSaveFileName(self, 'Save File', '', 'Images Files (*.png)')
        if filename:
            cv2.imwrite(filename, self.out_image)


    def aboutBox(self):
        QMessageBox.about(self, "About my Application",
                          "First line....\n"
                          "Second line.....\n"
                          "Third line.\n"
                          "\t Tab text...")

    def printImage(self):
        printer = QPrinter(QPrinter.HighResolution)
        dialog = QPrintDialog(printer, self)

        if dialog.exec_() == QPrintDialog.Accepted:
            self.ui.lblInputImg.print(printer)

    def showGammaWindow(self):
        self.gWindow = GammaWindow(parent=self)

    def DisplayImage(self, image=None):
        """Display the image with the current display-only zoom factor."""
        if image is not None and image.any():
            if isinstance(image, np.ndarray):
                # Convert to QImage for display
                if len(image.shape) == 3:  # RGB Image
                    qformat = QImage.Format_RGB888 if image.shape[2] == 3 else QImage.Format_RGBA8888
                else:  # Grayscale Image
                    qformat = QImage.Format_Indexed8

                h, w = image.shape[:2]
                # Convert NumPy array to QImage
                qt_image = QImage(image.data, w, h, image.strides[0], qformat)

                # Apply display-only zoom factor
                scaled_width = int(qt_image.width() * self.display_scale_factor)
                scaled_height = int(qt_image.height() * self.display_scale_factor)
                pixmap = QPixmap.fromImage(qt_image).scaled(
                    scaled_width, scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )

                # Display the pixmap in the label without altering label or window size
                self.ui.lblInputImg.setPixmap(pixmap)
                print(f"[DEBUG] Displaying with display_scale_factor: {self.display_scale_factor}")

    def eventFilter(self, obj, event):
        # We want to resize the image when the main window is resized to fit the new size of the label
        if event.type() == QtCore.QEvent.Resize and self.bImgloaded:
            self.ui.lblInputImg.setMinimumSize(1, 1)
            p = self.qt_image.scaled(self.ui.lblInputImg.width(), self.ui.lblInputImg.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.pix_image = QPixmap.fromImage(p)
            self.ui.lblInputImg.setPixmap(self.pix_image)
            self.add_to_history(self.out_image)  # Add modified image to history
            self.update_undo_redo_buttons()

        return super().eventFilter(obj, event)



    def start_edge_detection(self):
        """Start the edge detection in a separate thread."""
        if self.out_image is None:
            QMessageBox.warning(self, "No Image Loaded", "Please load an image first.")
            return

        # Start edge detection in a new thread
        self.edge_thread = EdgeDetectionThread(self.out_image)
        self.edge_thread.edge_detected.connect(self.display_edge_detected_image)
        self.edge_thread.start()

    def display_edge_detected_image(self, edges):
        """Display the edge-detected image."""
        # Display the processed edges image
        self.DisplayImage(image=edges)
        self.out_image = edges  # Update the current image with the edge-detected image

        # Add the result to history for Undo/Redo functionality
        self.add_to_history(self.out_image)
    def convGrayscale(self):
        self.out_image = cv2.cvtColor(self.out_image, cv2.COLOR_RGB2GRAY)
        self.DisplayImage(image=self.out_image)
        self.add_to_history(self.out_image)  # Add modified image to history
        self.update_undo_redo_buttons()

    def apply_sepia(self):
        # Define sepia filter matrix
        sepia_filter = np.array([[0.393, 0.769, 0.189],
                                 [0.349, 0.686, 0.168],
                                 [0.272, 0.534, 0.131]])
        # Apply sepia filter
        sepia_image = cv2.transform(self.out_image, sepia_filter)
        self.out_image = np.clip(sepia_image, 0, 255).astype(np.uint8)  # Clip values to be in valid range
        self.DisplayImage(image=self.out_image)
        self.add_to_history(self.out_image)  # Add modified image to history
        self.update_undo_redo_buttons()

    def apply_sharpening(self):
        # Define sharpening kernel
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        # Apply filter
        self.out_image = cv2.filter2D(self.out_image, -1, kernel)
        self.DisplayImage(image=self.out_image)
        self.add_to_history(self.out_image)  # Add modified image to history
        self.update_undo_redo_buttons()

    def convBGR(self):
        self.out_image = cv2.cvtColor(self.out_image, cv2.COLOR_RGB2BGR)
        self.DisplayImage(image=self.out_image)
        self.add_to_history(self.out_image)  # Add modified image to history
        self.update_undo_redo_buttons()

    def convRGB(self):
        self.out_image = cv2.cvtColor(self.out_image, cv2.COLOR_BGR2RGB)
        self.DisplayImage(image=self.out_image)
        self.add_to_history(self.out_image)  # Add modified image to history
        self.update_undo_redo_buttons()

    def convNegative(self):
        self.out_image = ~self.out_image
        self.DisplayImage(image=self.out_image)
        self.add_to_history(self.out_image)  # Add modified image to history
        self.update_undo_redo_buttons()

    def medianFilter(self):
        self.out_image = cv2.medianBlur(self.out_image, 5) # Example with a value of 5, could be a variable instead
        self.DisplayImage(image=self.out_image)
        self.add_to_history(self.out_image)  # Add modified image to history
        self.update_undo_redo_buttons()

    def gammaSlider(self, gamma):
        image = self.out_image
        gamma = gamma * 0.1
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        self.out_image = cv2.LUT(image, table)
        self.DisplayImage(image=self.out_image)
        self.add_to_history(self.out_image)  # Add modified image to history
        self.update_undo_redo_buttons()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # To avoid weird behaviors (smaller items, ...) on big resolution screens
    app.setStyle("fusion")
    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())
