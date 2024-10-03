//hier zijn alle libraries
#include <SPI.h>
#include <SD.h>
#include <Adafruit_Sensor.h>
#include <DHT.h>
#include <DHT_U.h>
// pinnen definen voor DHT sensoren
#define DHT11_PIN1 A2 // de eerste dht pin voor kant 1
#define DHTTYPE1    DHT11
#define DHT11_PIN2 2 // de tweede dht pin voor kant 2
#define DHTTYPE2    DHT11

DHT dht1(DHT11_PIN1, DHTTYPE1);
DHT dht2(DHT11_PIN2, DHTTYPE2);

File myFile;
const int chipSelect = 4; // dit is voor de SD kaart

//variabelen voor kant 1
const int vochtigheidsmeter1 = A0;
const int lichtdetector1 = A1;
const int temperat_1 = A2;
//int lichtwaarde1 = 0; // handiger voor code later

//variabelen voor kant 2
const int vochtigheidsmeter2 = A5;
const int lichtdetector2 = A4;
const int temperat_2 = A3;
//int lichtwaarde2 = 0;
int Pycharmcommand = Serial.read(); // wat wordt er gestuurd vanaf de pycharm code
void setup(){
  Serial.begin(9600);
  dht1.begin(); // start werking DHT-sensor
  dht2.begin();
 // if (!SD.begin(chipSelect)) {
 //   Serial.println("Card failed, or not present");
    // don't do anything more:
 //   while (1);
 // }
  //Serial.println("card initialized.");
}

void loop (void){
  if (Serial.available() > 0){
    if (Pycharmcommand = "Initiate sensorsimport Arduino 3"){
    

    //kant 1
    //vochtigheidsmeter1
    int aardvochtValue1 = analogRead(A0);
    //Serial.print("Aardevocht1 is: ");
    Serial.print(aardvochtValue1);
    delay(100); // anders gaat de code te snel en kunnen er mogelijk foute waardes uitkomen
    //lichtdetector1
    int lichtwaarde1 = analogRead(A1);
    //Serial.print("Lichtwaarde1 = ");
    Serial.println(lichtwaarde1);
    delay(1000);
    //temperat_1
    float humid_1 = dht1.readHumidity();
    float temp_1 = dht1.readTemperature();
    //Serial.print("Luchtvochtigheid1 = ");
    Serial.print(humid_1);
    //Serial.print("Temperatuur1 = ");
    Serial.print(temp_1);
    delay(1000);

    //kant 2
    //vochtigheidsmeter 2
    int aardvochtValue2 = analogRead(A5);
    //Serial.print("Aardevocht2 is: ");
    Serial.print(aardvochtValue2);
    delay(1000); // anders gaat de code te snel en kunnen er mogelijk foute waardes uitkomen
    //lichtdetector1
    int lichtwaarde2  = analogRead(A4);
    //Serial.print("Lichtwaarde2 = ");
    Serial.print(lichtwaarde2);
    delay(1000);
    //temperat_1
    float humid_2 = dht2.readHumidity();
    float temp_2 = dht2.readTemperature();
    //Serial.print("Luchtvochtigheid2 = ");
    Serial.print(humid_2);
    //Serial.print("Temperatuur2 = ");
    Serial.print(temp_2);
    delay(1000);

    //SD kaart


    String dataString = "";
    dataString += String(aardvochtValue1);  // Voeg aardevochtigheid toe
    dataString += String(temp_1);    // Voeg temperatuur toe
    dataString += String(humid_1);  // Voeg luchtvochtigheid toe
    dataString += String(lichtwaarde1);  // Voeg lichtwaarde1 toe
    dataString += String(aardvochtValue2);
    dataString += String(temp_2);
    dataString += String(humid_2);
    dataString += String(lichtwaarde2);

    File dataFile = SD.open("Arduino3/26/9.txt", FILE_WRITE);  // Open bestand op SD-kaart
    if (dataFile) {
      dataFile.println(dataString);
      //Serial.println("YASBITCH");
      dataFile.close();
    } else {
      //Serial.println("Kan bestand niet openen");
    dataFile.close();
    }
  if (Pycharmcommand = "Initiate SD-card Arduino 3"){
        // Lees het bestand regel voor regel
    while (dataFile.available()) {
      Serial.write(dataFile.read());
    }
    
    // Sluit het bestand
    dataFile.close();
  }
  }

}
}


