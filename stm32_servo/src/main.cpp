#include <Arduino.h>
#include <Wire.h>

#include <Servo.h>


Servo j0servo;
Servo j1servo;  // create servo object to control a servo
unsigned long prevMillis = 0;
unsigned int interval = 10;

int j0val, j1val;    // variable to read the value from the analog pin
bool j0dir, j1dir;
void setup() {
  j1servo.attach(PB1); 
  j0servo.attach(PA7);         // attaches the servo on pin 9 to the servo object
}

void loop() {
  if(millis() - prevMillis >= interval){
    if(j0val >= 180) j0dir = 0;
    if(j0val <= 0 && j0dir == 0) j0dir = 1;
    if(j0dir) j0val +=random(5);
    else j0val -=random(5);

    if(j1val >= 180) j1dir = 0;
    if(j1val <= 0 && j1dir == 0) j1dir = 1;
    if(j1dir) j1val +=random(5);
    else j1val -=random(5);

    prevMillis = millis();
    
  }
  j0servo.write(j0val);
  j1servo.write(j1val);

  
}