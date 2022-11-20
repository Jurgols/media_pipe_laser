import serial
import requests
import serial.tools.list_ports
#esp_addr = "10.109.6.173"

class SerialLine:
    def __init__(self, baud, fC, minChange=0.1, maxChange=0.5):
        self.baud = baud
        self.fC = fC
        self.last_val = [0 for i in range(fC)]
        self.minChange = minChange
        self.maxChange = maxChange
        
        ports=[]
        print('Looking for serial ports')
        while (len(ports) == 0):
            ports = serial.tools.list_ports.comports()
        ports = sorted(ports)
        if len(ports)==1:
            print(ports)
            self.port=ports[0]
        else:
            print(f'Choose serial port (0-{len(ports)}):')
            for i in range(len(ports)):
                print(f'[{i}]   {ports[i].device}')
            self.port = ports[int(input())]

        self.ser_dev = serial.Serial(port=self.port.device, baudrate=self.baud, timeout=1)
    def compVal(self, val):
        for i in range(self.fC):
            if abs(self.last_val[i]-val[i]) < self.minChange:
                val[i] = self.last_val[i]
            if abs(self.last_val[i]-val[i]) > self.maxChange:
                val[i] = (val[i]-self.last_val[i])*0.5+self.last_val[i]
        self.last_val = val
    def sendVal(self, val):
        self.ser_dev.write((0x00).to_bytes(1, byteorder='little'))
        for fing in val:
            self.ser_dev.write((1 if fing==0 else int(fing*255)).to_bytes(1, byteorder='little'))

        
#mod_fing = [int(fing * (180 - 1)/255 + 1) for fing in out_fing]
#print(mod_fing)
#r = requests.post(f'http://{esp_addr}/servoarr', json=mod_fing)
#print(r.status_code)