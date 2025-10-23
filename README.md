# EyePy - Psychophysiology Experiment Framework

A comprehensive Python framework for psychophysiology experiments including Pupil Light Reflex (PLR), pain threshold testing, eye tracking, and EEG recording.

## Features

- **Pupil Light Reflex (PLR)** measurements
- **Pain threshold testing** with Medoc thermal stimulator
- **Eye tracking** with Eyelink support
- **EEG recording** with parallel port triggers
- **Video/IR camera** recording
- **LED stimulation** control
- **Data logging** and analysis

## Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/eyepy.git
   cd eyepy
   ```

2. **Create virtual environment:**
   ```bash
   conda create -n eyepy python=3.9
   conda activate eyepy
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running Experiments

1. **Main experiment:**
   ```bash
   python main.py
   ```

2. **PLR experiment only:**
   ```bash
   python "PLR_MAIN_PAPER2 2.py"
   ```

3. **Demo mode:**
   - Select 'threshold_demo' or 'main_demo' in the GUI
   - Runs without hardware connections

## Hardware Support

- **Medoc Thermal Stimulator** (TCP/IP)
- **Eyelink Eye Tracker** (USB/Ethernet)
- **EEG System** (Parallel port)
- **Arduino LED** (Serial)
- **IR/Video Cameras** (USB)

## Configuration

Edit `config.py` to enable/disable hardware and adjust settings:

```python
# Hardware enable/disable
medoc_on = 1  # Enable Medoc
eyelink_on = 1  # Enable Eyelink
eeg_on = 1  # Enable EEG
led_on = True  # Enable LED

# Screen settings
scrsize = (1920, 1080)  # Adjust to your monitor
full_screen = True  # Set to False for windowed mode
```

## Data Output

The experiment creates:
- **Trials**: `{participant}_{session}_trials_{timestamp}.csv`
- **Events**: `{participant}_{session}_events_{timestamp}.csv`
- **Experiment Info**: `{participant}_{session}_exp_info_{timestamp}.json`
- **Eyelink**: `eyelink_{participant}_{session}.edf`
- **Video**: `VC_{participant}_{session}_{timestamp}.mp4`
- **IR**: `IR_{participant}_{session}_{timestamp}.avi`

## Project Structure

```
eyepy/
├── main.py                 # Main experiment script
├── config.py              # Configuration settings
├── dataRecordHandler.py   # Data logging and recording
├── drawingHandler.py      # Visual stimulus presentation
├── expirimentUtils.py     # Utility functions
├── eegHandler.py          # EEG interface
├── eyelinkHandler.py      # Eye tracking interface
├── MedocHandler.py        # Pain stimulator interface
├── LED.py                 # LED control
├── IRCamera.py           # IR camera interface
├── VideoCamera.py        # Video camera interface
├── flow.py               # Experiment flow control
├── log_files/            # Data output directory
└── requirements.txt      # Python dependencies
```

## Requirements

- **Python**: 3.8-3.11
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 50GB for data recording
- **Graphics**: Dedicated graphics card recommended

## Troubleshooting

### Common Issues

1. **PsychoPy Import Errors:**
   ```bash
   pip install --upgrade psychopy numpy
   ```

2. **Serial Port Issues:**
   ```bash
   python -c "import serial.tools.list_ports; print([p.device for p in serial.tools.list_ports.comports()])"
   ```

3. **Screen Resolution Issues:**
   - Update `scrsize` in `config.py`
   - Use `screeninfo` to detect available monitors

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues with:
- **PsychoPy**: https://psychopy.org/
- **Eyelink**: https://www.sr-research.com/
- **General Python**: https://docs.python.org/
