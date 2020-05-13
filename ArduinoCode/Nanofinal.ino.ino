// Include Libraries
#include "Arduino.h"
#include "SFE_BMP180.h"
#include "DS18B20.h"
#include "pulse-sensor-arduino.h"
#include "LiquidCrystal_PCF8574.h"


// Pin Definitions
#define DS18B20WP_PIN_DQ  2
#define GPS_PIN_TX  11
#define GPS_PIN_RX  10
#define HEARTPULSE_PIN_SIG  A0



// Global variables and defines
// There are several different versions of the LCD I2C adapter, each might have a different address.
// Try the given addresses by Un/commenting the following rows until LCD works follow the serial monitor prints. 
// To find your LCD address go to: http://playground.arduino.cc/Main/I2cScanner and run example.
#define LCD_ADDRESS 0x3F
//#define LCD_ADDRESS 0x27
// Define LCD characteristics
#define LCD_ROWS 4
#define LCD_COLUMNS 20
#define SCROLL_DELAY 150
#define BACKLIGHT 255
// object initialization
SFE_BMP180 bmp180;
DS18B20 ds18b20wp(DS18B20WP_PIN_DQ);
PulseSensor heartpulse;
LiquidCrystal_PCF8574 lcd20x4;


// define vars for testing menu
const int timeout = 10000;       //define timeout of 10 sec
char menuOption = 0;
long time0;

// Setup the essentials for your circuit to work. It runs first every time your circuit is powered with electricity.
void setup() 
{
    // Setup Serial which is useful for debugging
    // Use the Serial Monitor to view printed messages
    Serial.begin(9600);
    while (!Serial) ; // wait for serial port to connect. Needed for native USB
    Serial.println("start");
    
    //Initialize I2C device
    bmp180.begin();
    heartpulse.begin(HEARTPULSE_PIN_SIG);
    // initialize the lcd
    lcd20x4.begin(LCD_COLUMNS, LCD_ROWS, LCD_ADDRESS, BACKLIGHT); 
    menuOption = menu();
    
}

// Main logic of your circuit. It defines the interaction between the components you selected. After setup, it runs over and over again, in an eternal loop.
void loop() 
{
    
    
    if(menuOption == '1') {
    // BMP180 - Barometric Pressure, Temperature, Altitude Sensor - Test Code
    // Read Altitude from barometric sensor, note that the sensor is 1m accurate
    double bmp180Alt = bmp180.altitude();
    double bmp180Pressure = bmp180.getPressure();
    double bmp180TempC = bmp180.getTemperatureC();     //See also bmp180.getTemperatureF() for Fahrenheit
    Serial.print(F("Altitude: ")); Serial.print(bmp180Alt,1); Serial.print(F(" [m]"));
    Serial.print(F("\tpressure: ")); Serial.print(bmp180Pressure,1); Serial.print(F(" [hPa]"));
    Serial.print(F("\tTemperature: ")); Serial.print(bmp180TempC,1); Serial.println(F(" [°C]"));

    }
    else if(menuOption == '2') {
    // DS18B20 1-Wire Temperature Sensor - Waterproof - Test Code
    // Read DS18B20 temp sensor value in degrees celsius. for degrees fahrenheit use ds18b20wp.ReadTempF()
    float ds18b20wpTempC = ds18b20wp.readTempC();
    Serial.print(F("Temp: ")); Serial.print(ds18b20wpTempC); Serial.println(F(" [C]"));

    }
    else if(menuOption == '3')
    {
    // Disclaimer: The Ublox NEO-6M GPS Module is in testing and/or doesn't have code, therefore it may be buggy. Please be kind and report any bugs you may find.
    }
    else if(menuOption == '4') {
    // Heart Rate Pulse Sensor - Test Code
    //Measure Heart Rate
    int heartpulseBPM = heartpulse.BPM;
    Serial.println(heartpulseBPM);
    if (heartpulse.QS == true) {
    Serial.println("PULSE");
    heartpulse.QS = false;
    }
    }
    else if(menuOption == '5') {
    // LCD Display 20x4 I2C - Test Code
    // The LCD Screen will display the text of your choice.
    lcd20x4.clear();                          // Clear LCD screen.
    lcd20x4.selectLine(2);                    // Set cursor at the begining of line 2
    lcd20x4.print("    Circuito.io  ");                   // Print print String to LCD on first line
    lcd20x4.selectLine(3);                    // Set cursor at the begining of line 3
    lcd20x4.print("      Rocks!  ");                   // Print print String to LCD on second line
    delay(1000);

    }
    
    if (millis() - time0 > timeout)
    {
        menuOption = menu();
    }
    
}



// Menu function for selecting the components to be tested
// Follow serial monitor for instrcutions
char menu()
{

    Serial.println(F("\nWhich component would you like to test?"));
    Serial.println(F("(1) BMP180 - Barometric Pressure, Temperature, Altitude Sensor"));
    Serial.println(F("(2) DS18B20 1-Wire Temperature Sensor - Waterproof"));
    Serial.println(F("(3) Ublox NEO-6M GPS Module"));
    Serial.println(F("(4) Heart Rate Pulse Sensor"));
    Serial.println(F("(5) LCD Display 20x4 I2C"));
    Serial.println(F("(menu) send anything else or press on board reset button\n"));
    while (!Serial.available());

    // Read data from serial monitor if received
    while (Serial.available()) 
    {
        char c = Serial.read();
        if (isAlphaNumeric(c)) 
        {   
            
            if(c == '1') 
          Serial.println(F("Now Testing BMP180 - Barometric Pressure, Temperature, Altitude Sensor"));
        else if(c == '2') 
          Serial.println(F("Now Testing DS18B20 1-Wire Temperature Sensor - Waterproof"));
        else if(c == '3') 
          Serial.println(F("Now Testing Ublox NEO-6M GPS Module - note that this component doesn't have a test code"));
        else if(c == '4') 
          Serial.println(F("Now Testing Heart Rate Pulse Sensor"));
        else if(c == '5') 
          Serial.println(F("Now Testing LCD Display 20x4 I2C"));
            else
            {
                Serial.println(F("illegal input!"));
                return 0;
            }
            time0 = millis();
            return c;
        }
    }
}
