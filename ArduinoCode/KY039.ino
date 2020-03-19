//pulse monitor test script
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27,20,4);  // set the LCD address to 0x27 for a 16 chars and 2 line display
int ledPin = 13;
int sensorPin =A0;
double alpha = 0.75;
int period =3000;
double change =0.0;

void setup ()
{
  lcd.init();                      // initialize the lcd 
  // Print a message to the LCD.
  lcd.backlight();
  pinMode (ledPin, OUTPUT);
  Serial.begin(9600);
}

void loop ()
{
  static double oldValue =0;
  static double oldChange =0;
  int rawValue =analogRead (sensorPin);
  lcd.setCursor(0,0);
  lcd.print("Pulse Rate");
  if(rawValue>900)
  {
  Serial.println (rawValue-920);
  lcd.setCursor(0,1);
  lcd.print(rawValue-920);
  }
  else
  {
  Serial.println("0");
  lcd.setCursor(0,1);
  lcd.print("0");
  }
  delay (period);
}
