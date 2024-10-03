//libraries overzicht
#include <SPI.h>                // voor de SD kaart
#include <SD.h>                 // voor de SD kaart
#include <dht.h>                // voor de DHT11 sensor
#include <LiquidCrystal_I2C.h>  // voor het scherm
#include <Wire.h>               // voor het scherm
#include <TimeLib.h>           // voor de tid
const int PompPin = 2;          // voor de pomp
const int LichtSensorPin = A0;  // voor de lichtsensor
const int AVochtMeter = A1;     // de vochtmeter in de aarde

//Speciaal schermdingen
LiquidCrystal_I2C lcd(0x27, 16, 2);
const int LCDWidth = 16;
const int LCDHeight = 2;

#define DHT11_PIN 3  // speciale functie voor DHT11 sensor
dht DHT;

//SD-kaart waardes
File myFile;

#define TIME_MSG_LEN 11// time sync to PC is HEADER followed by Unix time_t as ten ASCII digits
#define TIME_HEADER 'T'// HEader tag for serial time sync message
#define TIME_Request 7 // ASCI bell character requests a time sync message
void setup() {
  Serial.begin(9600);
  pinMode(PompPin, OUTPUT);        // dit zorgt ervoor dat de computer weet dat de pomp een output is
  pinMode(LichtSensorPin, INPUT);  // computer weet dat dit input is
  lcd.begin();                     // start lcd scherm op
  lcd.backlight();                 // start backlight op

  //SD-kaart setup
  const int chipSelect = 10;
  //if (!SD.begin(chipSelect)) {
    //Serial.println("SD-kaart mislukt");
    //return;
 // }
  //Serial.println("SD-kaart geslaagd");


}

void loop() {
  //AardeVochtMeter
  int AVochtMeter_waarde = analogRead(A1);  // leest de waardes van de meter
  Serial.print(AVochtMeter_waarde);         // print de waarde

  //Pomp code
  digitalWrite(PompPin, HIGH);  // dit zet de pomp aan, moeten we nog even de juiste waardes voor vinden

  //DHT11 sensor
  int chk = DHT.read11(DHT11_PIN);           // leest de sensor
  int Temperatuur_waarde = DHT.temperature;  // dit is de temp
  Serial.print(DHT.temperature);
  int Vochtigheids_waarde = DHT.humidity;  // dit is de vochtigheidsgehalte van de lucht
  Serial.print(DHT.humidity);

  // Lichtsensor
  int LichtSensorPin_waarde = analogRead(A3);
  Serial.println(LichtSensorPin_waarde);  // zorgt voor een enter, zodat de pythoncode ook goed werkt

  // LCD-scherm
  delay(1000);             // even de computer laten rusten
  lcd.clear();             // wist het vorige scherm
  lcd.setCursor(0, 0);     // scherm begint op de bovenste rij
  lcd.print("VochtA = ");  // VochtA = aarde vochtigheid
  lcd.print(AVochtMeter_waarde);
  lcd.setCursor(0, 1);  // Scherm begint op tweede rij
  lcd.print("Licht = ");
  lcd.print(LichtSensorPin_waarde);
  delay(3000);  // dit is een goede tijd om het te lezen
  lcd.clear();  // we beginnen weer met een nieuw scherm
  lcd.setCursor(0, 0);
  lcd.print("Temp = ");
  lcd.print(Temperatuur_waarde);
  lcd.setCursor(0, 1);
  lcd.print("VochtL = ");  // staat voor de vochtigheid in de lucht
  lcd.print(Vochtigheids_waarde);
  delay(3000);

  //SD-kaart
  String dataString = "";
  dataString += String(AVochtMeter_waarde);     // voegt de vochtigheidswaarde toe van de aarde
  dataString += float(Temperatuur_waarde);     // voegt de temperatuur toe
  dataString += float(Vochtigheids_waarde);    // voegt de vochtigheidswaarde toe van de lucht
  dataString += String(LichtSensorPin_waarde);  // voegt het licht toe
  //Serial.println(dataString);

  File dataFile = SD.open("AS.txt", FILE_WRITE);  // opent bestand op de SD kaart
  if (dataFile) {
    dataFile.println(dataString);
    dataFile.close();
    //Serial.println("Data gelukt");
  } else {
    //Serial.println("Data mislukt");
  }
  delay(9000000);

}

