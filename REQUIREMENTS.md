# EyePy Experiment Setup Requirements

## Overview
This is a comprehensive psychophysiology experiment system that combines:
- **Pupil Light Reflex (PLR)** measurements
- **Pain threshold** testing with Medoc thermal stimulator
- **Eye tracking** with Eyelink
- **EEG** recording with parallel port triggers
- **Video/IR camera** recording
- **LED** stimulation control

## System Requirements

### Hardware Requirements
- **Computer**: Windows 10/11 or macOS (Linux supported but not tested)
- **RAM**: Minimum 8GB, recommended 16GB
- **Storage**: 50GB free space for data recording
- **Graphics**: Dedicated graphics card recommended for smooth stimulus presentation
- **Monitors**: 2 monitors recommended (one for experiment, one for monitoring)

### External Hardware (Optional)
- **Medoc Thermal Stimulator**: For pain threshold testing
- **Eyelink Eye Tracker**: For eye movement recording
- **EEG System**: With parallel port support
- **Arduino/Serial LED**: For light stimulation
- **IR Camera**: For pupil recording
- **Video Camera**: For behavioral recording

## Python Environment Setup

### 1. Install Python
```bash
# Download Python 3.8-3.11 from python.org
# OR use conda/miniconda (recommended)
```

### 2. Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n eyepy python=3.9
conda activate eyepy

# OR using venv
python -m venv eyepy_env
# Windows:
eyepy_env\Scripts\activate
# macOS/Linux:
source eyepy_env/bin/activate
```

## Required Python Packages

### Core Dependencies
```bash
pip install psychopy>=2023.2.0
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install matplotlib>=3.5.0
pip install scipy>=1.7.0
```

### Hardware Interface Dependencies
```bash
# Serial communication (LED control)
pip install pyserial>=3.5

# Screen/monitor detection
pip install screeninfo>=0.8.0

# Mouse control (for VAS)
pip install pynput>=1.7.0

# Parallel port (EEG triggers) - Windows only
pip install pyparallel>=1.0.0
# OR for cross-platform:
pip install parallel>=1.0.0
```

### Eye Tracking Dependencies
```bash
# Eyelink support
pip install pylink>=1.0.0

# Alternative eye tracking (if not using Eyelink)
pip install opencv-python>=4.5.0
pip install mediapipe>=0.8.0
```

### Data Processing Dependencies
```bash
# Data analysis
pip install scikit-learn>=1.0.0
pip install seaborn>=0.11.0

# File handling
pip install pathlib2>=2.3.0  # For older Python versions
```

### Optional Dependencies
```bash
# Video processing
pip install opencv-python>=4.5.0
pip install imageio>=2.9.0

# Network communication (Medoc)
# Built-in socket module, no additional install needed

# GUI enhancements
pip install tkinter  # Usually included with Python
```

## Complete Installation Script

### Windows (PowerShell)
```powershell
# Create environment
conda create -n eyepy python=3.9 -y
conda activate eyepy

# Install packages
pip install psychopy>=2023.2.0
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install matplotlib>=3.5.0
pip install scipy>=1.7.0
pip install pyserial>=3.5
pip install screeninfo>=0.8.0
pip install pynput>=1.7.0
pip install pylink>=1.0.0
pip install opencv-python>=4.5.0
pip install scikit-learn>=1.0.0
pip install seaborn>=0.11.0
```

### macOS/Linux
```bash
# Create environment
conda create -n eyepy python=3.9 -y
conda activate eyepy

# Install packages
pip install psychopy>=2023.2.0
pip install numpy>=1.21.0
pip install pandas>=1.3.0
pip install matplotlib>=3.5.0
pip install scipy>=1.7.0
pip install pyserial>=3.5
pip install screeninfo>=0.8.0
pip install pynput>=1.7.0
pip install pylink>=1.0.0
pip install opencv-python>=4.5.0
pip install scikit-learn>=1.0.0
pip install seaborn>=0.11.0
```

## Hardware Setup Instructions

### 1. Medoc Thermal Stimulator
- **Connection**: TCP/IP network connection
- **IP Address**: Configure in `config.py` (default: 10.101.119.124)
- **Port**: 20121
- **Setup**: Ensure Medoc is on same network as experiment computer

### 2. Eyelink Eye Tracker
- **Software**: Install Eyelink SDK
- **Connection**: USB or Ethernet
- **Calibration**: Run calibration before each session
- **Data**: EDF files saved automatically

### 3. EEG System
- **Connection**: Parallel port (LPT1) or USB-to-parallel adapter
- **Address**: Configure in `config.py` (default: 0x03EFC)
- **Triggers**: 8-bit trigger codes sent for events

### 4. LED Stimulation
- **Hardware**: Arduino or serial-controlled LED
- **Connection**: USB serial port
- **Baud Rate**: 9800 (configurable)
- **Protocol**: 'H' for high, 'L' for low

### 5. Cameras
- **IR Camera**: For pupil recording
- **Video Camera**: For behavioral recording
- **Connection**: USB or network
- **Recording**: Synchronized with experiment events

## Configuration

### 1. Edit `config.py`
```python
# Screen settings
scrsize = (1920, 1080)  # Adjust to your monitor
full_screen = True  # Set to False for windowed mode

# Hardware enable/disable
medoc_on = 1  # Enable Medoc
eyelink_on = 1  # Enable Eyelink
eeg_on = 1  # Enable EEG
led_on = True  # Enable LED
IR_camera_on = 1  # Enable IR camera
video_camera_on = 1  # Enable video camera

# Network settings
medoc_host = '10.101.119.124'  # Medoc IP address
medoc_port = 20121

# EEG settings
eeg_port_address = 0x03EFC  # Parallel port address
```

### 2. Create Data Directory
```bash
mkdir log_files
```

## Running the Experiment

### 1. Main Experiment
```bash
python main.py
```

### 2. PLR Experiment Only
```bash
python "PLR_MAIN_PAPER2 2.py"
```

### 3. Demo Mode
- Set session to 'threshold_demo' or 'main_demo' in GUI
- Runs without hardware connections

## Troubleshooting

### Common Issues

1. **PsychoPy Import Errors**
   ```bash
   pip install --upgrade psychopy
   pip install --upgrade numpy
   ```

2. **Serial Port Issues**
   ```bash
   # Check available ports
   python -c "import serial.tools.list_ports; print([p.device for p in serial.tools.list_ports.comports()])"
   ```

3. **Parallel Port Issues (Windows)**
   ```bash
   # Install parallel port driver
   pip install pyparallel
   # OR use USB-to-parallel adapter
   ```

4. **Eyelink Connection Issues**
   - Ensure Eyelink software is running
   - Check network connection
   - Verify IP settings

5. **Screen Resolution Issues**
   - Update `scrsize` in `config.py`
   - Use `screeninfo` to detect available monitors

### Performance Optimization

1. **Close unnecessary applications**
2. **Use dedicated graphics card**
3. **Disable Windows updates during experiments**
4. **Use SSD for data storage**

## Data Output

The experiment creates the following data files:
- **Trials**: `{participant}_{session}_trials_{timestamp}.csv`
- **Events**: `{participant}_{session}_events_{timestamp}.csv`
- **Experiment Info**: `{participant}_{session}_exp_info_{timestamp}.json`
- **Eyelink**: `eyelink_{participant}_{session}.edf`
- **Video**: `VC_{participant}_{session}_{timestamp}.mp4`
- **IR**: `IR_{participant}_{session}_{timestamp}.avi`

## Support

For issues with:
- **PsychoPy**: https://psychopy.org/
- **Eyelink**: https://www.sr-research.com/
- **Medoc**: Contact Medoc support
- **General Python**: https://docs.python.org/

## Version Compatibility

- **Python**: 3.8-3.11
- **PsychoPy**: 2023.2.0+
- **NumPy**: 1.21.0+
- **Pandas**: 1.3.0+

## Notes

- Always test in demo mode first
- Ensure all hardware is connected before running
- Backup data regularly
- Check logs for any error messages
- Use consistent participant IDs

